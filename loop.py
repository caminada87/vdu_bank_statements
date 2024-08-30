import logging, pathlib, time, pyodbc, os, math, datetime
from importlib.machinery import SourceFileLoader

import pytesseract, json, torch, re
from transformers import LayoutLMv2ForTokenClassification
from transformers import LayoutLMv2Processor
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('VDU_bank_statements')
logger.info('Logger created')

os.environ['HTTP_PROXY'] = 'http://autoproxy.ktag.ch:8080'
os.environ['HTTPS_PROXY'] = 'http://autoproxy.ktag.ch:8080'
os.environ["CURL_CA_BUNDLE"]=""
logger.info('Proxy is set')

config_path = os.path.join(pathlib.Path().resolve(), 'instance', 'config.py')
logger.info(f"Config path: {config_path}")
config_module= SourceFileLoader("config",config_path).load_module()
config = config_module.Config()

label_list = ['other', 'iban_key', 'iban', 'abschlussdatum_key', 'abschlussdatum', 'zinsertrag_key', 'zinsertrag', 'steuerwert_key', 'steuerwert', 'waehrung_key', 'waehrung', 'startdatum_key', 'startdatum', 'zinsaufwand_key', 'zinsaufwand', 'vst_key', 'vst', 'spesen_key', 'spesen', 'zinsertrag_vstpfl_key', 'zinsertrag_vstpfl', 'abzug_key', 'abzug']
label_list.sort()

id2label = {v: k for v, k in enumerate(label_list)}
label2id = {k: v for v, k in enumerate(label_list)}
logger.info('label list and dictionarys created.')

#ssl._create_default_https_context = ssl._create_unverified_context

model = LayoutLMv2ForTokenClassification.from_pretrained("./model", num_labels=len(label_list))
logger.info('model loaded')
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased" ,revision="no_ocr")
logger.info('processor loaded')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('device set to: {}'.format(device))
model.to(device)
logger.info('model pushed to device.')

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def getPred(encoded_inputs):
    labels = encoded_inputs.pop('labels').squeeze().tolist()

    for k,v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)

    outputs = model(**encoded_inputs)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()

    width, height = image.size

    true_predictions = [id2label[prediction] for prediction, label in zip(predictions, labels) if label != -100]
    true_labels = [id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
    true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]


    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    label2color = { 'other':'black', 
                    'iban_key': 'brown', 
                    'iban':'brown', 
                    'abschlussdatum_key':'blue', 
                    'abschlussdatum': 'blue', 
                    'zinsertrag_key':'violet', 
                    'zinsertrag':'violet',
                    'steuerwert_key': 'green', 
                    'steuerwert':'green', 
                    'waehrung_key':'orange', 
                    'waehrung': 'orange',
                    "startdatum_key": 'aqua',
                    "startdatum": 'aqua',
                    "zinsaufwand_key": 'FireBrick',
                    "zinsaufwand": 'FireBrick',
                    "vst_key": 'Gold',
                    "vst": 'Gold',
                    "spesen_key": 'SaddleBrown',
                    "spesen": 'SaddleBrown',
                    "zinsertrag_vstpfl_key": 'YellowGreen',
                    "zinsertrag_vstpfl": 'YellowGreen',
                    "abzug_key": 'DeepPink',
                    "abzug": 'DeepPink'
                }

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = prediction
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)
    
    return (predictions, token_boxes, image)

def get_plaintext_boxcontents_and_coordinates(label):
    indices = [index for index, element in enumerate(predictions) if element == label2id[label]]
    #print(indices)
    bboxes = [bbox for index, bbox in enumerate(token_boxes) if index in indices]

    #print(bboxes)
    bboxes_unique = []
    for bbox in bboxes:
        if bbox not in bboxes_unique:
            bboxes_unique.append(bbox)
    #print(bboxes_unique)
    plaintext_boxcontents = []

    for bbox in bboxes_unique:
        indices = [index for index, element in enumerate(encoded_inputs['bbox'][0].tolist()) if element == bbox]
        #print(indices)
        ids = [encoded_inputs['input_ids'][0].tolist()[index] for index in indices]
        #print(ids)
        plaintext_boxcontents.append({'content': processor.tokenizer.decode(ids), 'coordinates': bbox})

    return plaintext_boxcontents
    #bbox = [x1,y1,x2,y2]

def get_iban_list():
    plaintext_boxcontents = get_plaintext_boxcontents_and_coordinates('iban')

    ibans = []
    current_iban = ''
    current_iban_x1_y1 = []
    current_iban_x2_y2 = []
    regex_iban_start = re.compile(r"[a-z]{2}[0-9]{2}")
    regex_non_alphanumeric = re.compile(r"[^a-z0-9]*")

    for pbc in plaintext_boxcontents:
        #Filter Non Alphanumeric
        content = pbc['content'].lower()
        content = regex_non_alphanumeric.sub('', content)
        #sometimes the key 'iban' is mistaken for being part of the actual iban
        content.replace('iban', '')
        if len(content) > 0:
            #check for an IBAN start:
            res = regex_iban_start.search(content)
            if not res is None:
                #IBAN starts!
                if current_iban != '':
                    #Check if previous IBAN is valid and ADD or is invalid and overwrite
                    #Swiss IBAN: 21 characters. Shortest (Norway 15C). Longest (Russia 33C)
                    if (current_iban.startswith('ch') and len(current_iban) == 21) or (len(current_iban) >= 15 and len(current_iban) <= 33):
                        #valid
                        ibans.append({'iban': current_iban, 'coordinates': current_iban_x1_y1 + current_iban_x2_y2})
                    current_iban = ''
                #IBAN Started:
                current_iban = content[res.span()[0]:len(content)]
                current_iban_x1_y1 = pbc['coordinates'][0:2]
                current_iban_x2_y2 = pbc['coordinates'][2:4]
            else:
                if len(current_iban) > 0:
                    current_iban += content
                    current_iban_x2_y2 = pbc['coordinates'][2:4]
        #print(len(current_iban))
        #print(current_iban)
    if (current_iban.startswith('ch') and len(current_iban) == 21) or (len(current_iban) >= 15 and len(current_iban) <= 33):
        #valid
        ibans.append({'iban': current_iban, 'coordinates': current_iban_x1_y1 + current_iban_x2_y2})
    return ibans

def get_number_list(label):
    if label not in list(id2label.values()):
        return []
    plaintext_boxcontents = get_plaintext_boxcontents_and_coordinates(label)
    numbers = []
    current_number = ''
    current_number_x1_y1 = []
    current_number_x2_y2 = []
    regex_non_numeric_non_decimalseparator = re.compile(r"[^0-9.,]*")
    valid_number = re.compile(r"^[0-9]+\.[0-9]{2}$")

    for pbc in plaintext_boxcontents:
        #Filter Non Alphanumeric
        content = pbc['content'].lower()
        content = regex_non_numeric_non_decimalseparator.sub('', content)
        content = content.replace(',','.') # replace all commas by dots
        content = content.replace('.', '', content.count('.')-1) # remove all dots but last one
        if len(content) > 0:
            #print('current_number: ' + str(current_number))
            if current_number != '':
                #If that is the case: same number splitted! Add!
                #ABS(x1 (now) - x2 (previous)) <= 10
                #ABS(y1 (now) - y1 (previous)) < 5
                x1_now = int(pbc['coordinates'][0])
                x2_previous = int(current_number_x2_y2[0])
                absolute_horizontal_space = abs(x1_now - x2_previous)
                y1_now = pbc['coordinates'][1]
                y1_previous = current_number_x1_y1[1]
                absolute_vertical_space = abs(y1_now - y1_previous)

                if (abs(absolute_horizontal_space) <= 10) and (abs(absolute_vertical_space) <= 5):
                    current_number += content
                    current_number_x2_y2 = pbc['coordinates'][2:4]
                    continue
                else: 
                    #If there are no decimals, add them
                    if current_number.count('.') == 0:
                        current_number += '.00'
                    #Valid?
                    if valid_number.match(current_number):
                        numbers.append({label: current_number, 'coordinates': current_number_x1_y1 + current_number_x2_y2})

                    current_number = ''

            current_number = content
            current_number_x1_y1 = pbc['coordinates'][0:2]
            current_number_x2_y2 = pbc['coordinates'][2:4]
            #print(current_number)
    #If there are no decimals, add them
    if current_number.count('.') == 0:
        current_number += '.00'
    #Valid?
    if valid_number.match(current_number):
        numbers.append({label: current_number, 'coordinates': current_number_x1_y1 + current_number_x2_y2})

    return numbers

def get_date_list(label):
    if label not in list(id2label.values()):
        return []
    plaintext_boxcontents = get_plaintext_boxcontents_and_coordinates(label)
    dates = []
    current_date = ''
    current_date_x1_y1 = []
    current_date_x2_y2 = []
    regex_non_numeric_non_separator = re.compile(r"[^0-9.,]*")
    valid_date = re.compile(r"^[0-9]{1,2}\.[0-9]{1,2}\.([0-9]{2}|[0-9]{4})$")

    for pbc in plaintext_boxcontents:
        #Filter Non Alphanumeric
        content = pbc['content'].lower()
        content = regex_non_numeric_non_separator.sub('', content)
        content = content.replace(',','.') # replace all commas by dots
        content = content.replace('.', '', content.count('.')-2) # remove all dots but last two
        if len(content) > 0:
            if current_date != '':
                #If that is the case: same number splitted! Add!
                #ABS(x1 (now) - x2 (previous)) <= 10
                #ABS(y1 (now) - y1 (previous)) < 5
                x1_now = int(pbc['coordinates'][0])
                x2_previous = int(current_date_x2_y2[0])
                absolute_horizontal_space = abs(x1_now - x2_previous)
                y1_now = pbc['coordinates'][1]
                y1_previous = current_date_x1_y1[1]
                absolute_vertical_space = abs(y1_now - y1_previous)

                if (abs(absolute_horizontal_space) <= 10) and (abs(absolute_vertical_space) <= 5):
                    current_date += content
                    current_date_x2_y2 = pbc['coordinates'][2:4]
                    continue
                else: 
                    #Valid?
                    if valid_date.match(current_date):
                        dates.append({label: current_date, 'coordinates': current_date_x1_y1 + current_date_x2_y2})

                    current_date = ''

            current_date = content
            current_date_x1_y1 = pbc['coordinates'][0:2]
            current_date_x2_y2 = pbc['coordinates'][2:4]
            #print(current_date)
    #Valid?
    if valid_date.match(current_date):
        dates.append({label: current_date, 'coordinates': current_date_x1_y1 + current_date_x2_y2})

    return dates

def getDocumentInfos(img):
    image = Image.open(img).convert("RGB")
    boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='deu')
    #Prepare pytesseract output for model usage...
    resultText = ''
    count = 0
    for i in range(len(boxes['level'])):
        if boxes['text'][i].strip() != '':
            rel_left = boxes['left'][i]/image.size[0]
            rel_top = boxes['top'][i]/image.size[1]
            rel_width = boxes['width'][i]/image.size[0]
            rel_height = boxes['height'][i]/image.size[1]
            conf = boxes['conf'][i] #confidence -1 | 0-100
            block_num = boxes['block_num'][i] #box number
            par_num = boxes['par_num'][i] #paragraph number
            line_num = boxes['line_num'][i] #line number
            word_num = boxes['word_num'][i] #word number
            level = boxes['level'][i] #level

            word = {'id':count,
                    'text':boxes['text'][i].strip(),
                    'left':rel_left,
                    'top':rel_top,
                    'width':rel_width,
                    'height':rel_height,
                    'conf':conf,
                    'block_num':block_num,
                    'par_num':par_num,
                    'line_num':line_num,
                    'word_num':word_num,
                    'level':level
                    }
            #if word['conf'] > 20:
            resultText += (f'{json.dumps(word)},')
            count = count + 1

    resultText = '{"labels":[' + resultText.rstrip(',') + ']}'
    
    ocr = json.loads(resultText)

    words = []
    bboxes = []

    for label in ocr['labels']:
        words.append(label['text'])
        x1 = round(float(label["left"]) * 1000)
        y1 = round(float(label["top"]) * 1000)
        x2 = round((float(label["left"]) * 1000) + (float(label["width"]) * 1000))
        y2 = round((float(label["top"]) * 1000) + (float(label["height"]) * 1000))
        bbox = [x1,y1,x2,y2]
        bboxes.append(bbox)

    ner_tag_dummies = [0] * len(bboxes)
    encoded_inputs = processor(image, words, boxes=bboxes, word_labels=ner_tag_dummies, padding="max_length", truncation=True, return_tensors="pt")
    
    unused_labels = encoded_inputs.pop('labels').squeeze().tolist()
    for k,v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)
    
    outputs = model(**encoded_inputs)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()

    list_startdatum = get_date_list('startdatum', predictions, token_boxes, encoded_inputs)
    list_abschlussdatum = get_date_list('abschlussdatum', predictions, token_boxes, encoded_inputs)
    list_steuerwert = get_number_list('steuerwert')
    list_zinsertrag = get_number_list('zinsertrag', predictions, token_boxes, encoded_inputs)
    list_spesen = get_number_list('spesen', predictions, token_boxes, encoded_inputs)
    list_iban = get_iban_list(predictions, token_boxes, encoded_inputs)

    logger.info('IBANS')
    logger.info(list_iban)
    logger.info(list_steuerwert)

def get_nearest_box_index(box, listOfBoxes, force_y_larger = False, force_y_smaller = False, box_ref_smaller = {}):
    if len(listOfBoxes) == 0:
        return -1
    if len(listOfBoxes) == 1:
        return 0

    boxCenterX = (box['coordinates'][2] + box['coordinates'][0]) / 2
    boxCenterY = (box['coordinates'][3] + box['coordinates'][1]) / 2
    refBoxCenterY = 0
    if not box_ref_smaller == {}: refBoxCenterY = (box_ref_smaller['coordinates'][3] + box_ref_smaller['coordinates'][1]) / 2

    min_distance = -1
    min_index = -1
    index = 0

    for listBox in listOfBoxes:
        #print(listBox['content'])
        listBoxCenterX = (listBox['coordinates'][2] + listBox['coordinates'][0]) / 2
        listBoxCenterY = (listBox['coordinates'][3] + listBox['coordinates'][1]) / 2
        distance = math.sqrt(math.pow(boxCenterX-listBoxCenterX,2) + math.pow(boxCenterY-listBoxCenterY, 2))
        #print('x1: {}, x2: {}, y1: {}, y2: {}'.format(boxCenterX, listBoxCenterX, boxCenterY, listBoxCenterY))
        #print(distance)
        if min_distance == -1 or distance < min_distance:
            if (not force_y_larger or (force_y_larger and (listBoxCenterY-boxCenterY) > 0)):
                if (not force_y_smaller or (force_y_smaller and (refBoxCenterY - listBoxCenterY) > 0)):
                    min_distance = distance
                    min_index = index
        index += 1
    
    return min_index

def get_bank_account_list():
    bank_account_list = []

    iban_list = get_iban_list()

    if len(iban_list) > 0:
        multi_iban_force_y_larger = False
        if len(iban_list) > 1:
            multi_iban_force_y_larger = True

        steuerwert_list = get_number_list('steuerwert')
        zinsertrag_list = get_number_list('zinsertrag')
        zinsertrag_vstpfl_list = get_number_list('zinsertrag_vstpfl')
        zinsaufwand_list = get_number_list('zinsaufwand')
        waehrung_list = get_plaintext_boxcontents_and_coordinates('waehrung')
        startdatum_list = get_date_list('startdatum')
        abschlussdatum_list = get_date_list('abschlussdatum')
        spesen_list = get_number_list('spesen')
        vst_list = get_number_list('vst')
        abzug_list = get_number_list('abzug')

        for iban in iban_list:
            multi_iban_force_y_smaller = False
            box_ref_smaller = {}
            current_iban_index = iban_list.index(iban)
            if current_iban_index < (len(iban_list) - 1):
                multi_iban_force_y_smaller = True
                box_ref_smaller = iban_list[current_iban_index+1]

            current_bank_account = {'iban': iban['iban']}

            nearest_startdatum_index = get_nearest_box_index(iban, startdatum_list)
            nearest_abschlussdatum_index = get_nearest_box_index(iban, abschlussdatum_list)

            nearest_steuerwert_index = get_nearest_box_index(iban, steuerwert_list, multi_iban_force_y_larger, multi_iban_force_y_smaller, box_ref_smaller)
            if nearest_steuerwert_index != -1 and len(waehrung_list) > 0:
                steuerwert_waehrung_index = get_nearest_box_index(steuerwert_list[nearest_steuerwert_index], waehrung_list)
            else:
                steuerwert_waehrung_index = -1

            nearest_zinsertrag_index = get_nearest_box_index(iban, zinsertrag_list, multi_iban_force_y_larger, multi_iban_force_y_smaller, box_ref_smaller)
            if nearest_zinsertrag_index != -1 and len(waehrung_list) > 0:
                zinsertrag_waehrung_index = get_nearest_box_index(zinsertrag_list[nearest_zinsertrag_index], waehrung_list)
            else:
                zinsertrag_waehrung_index = -1
            
            nearest_zinsertrag_vstpfl_index = get_nearest_box_index(iban, zinsertrag_vstpfl_list, multi_iban_force_y_larger, multi_iban_force_y_smaller, box_ref_smaller)
            if nearest_zinsertrag_vstpfl_index != -1 and len(waehrung_list) > 0:
                zinsertrag_vstpfl_waehrung_index = get_nearest_box_index(zinsertrag_vstpfl_list[nearest_zinsertrag_vstpfl_index], waehrung_list)
            else:
                zinsertrag_vstpfl_waehrung_index = -1

            nearest_zinsaufwand_index = get_nearest_box_index(iban, zinsaufwand_list, multi_iban_force_y_larger, multi_iban_force_y_smaller, box_ref_smaller)
            if nearest_zinsaufwand_index != -1 and len(waehrung_list) > 0:
                zinsaufwand_waehrung_index = get_nearest_box_index(zinsaufwand_list[nearest_zinsaufwand_index], waehrung_list)
            else:
                zinsaufwand_waehrung_index = -1

            nearest_spesen_index = get_nearest_box_index(iban, spesen_list, multi_iban_force_y_larger, multi_iban_force_y_smaller, box_ref_smaller)
            if nearest_spesen_index != -1 and len(waehrung_list) > 0:
                spesen_waehrung_index = get_nearest_box_index(spesen_list[nearest_spesen_index], waehrung_list)
            else:
                spesen_waehrung_index = -1

            nearest_vst_index = get_nearest_box_index(iban, vst_list, multi_iban_force_y_larger, multi_iban_force_y_smaller, box_ref_smaller)
            if nearest_vst_index != -1 and len(waehrung_list) > 0:
                vst_waehrung_index = get_nearest_box_index(vst_list[nearest_vst_index], waehrung_list)
            else:
                vst_waehrung_index = -1

            nearest_abzug_index = get_nearest_box_index(iban, abzug_list, multi_iban_force_y_larger, multi_iban_force_y_smaller, box_ref_smaller)
            if nearest_abzug_index != -1 and len(waehrung_list) > 0:
                abzug_waehrung_index = get_nearest_box_index(abzug_list[nearest_abzug_index], waehrung_list)
            else:
                abzug_waehrung_index = -1
            
            if nearest_startdatum_index != -1: current_bank_account['startdatum'] = startdatum_list[nearest_startdatum_index]['startdatum']
            if nearest_abschlussdatum_index != -1: current_bank_account['abschlussdatum'] = abschlussdatum_list[nearest_abschlussdatum_index]['abschlussdatum']
            if nearest_steuerwert_index != -1: current_bank_account['steuerwert'] = steuerwert_list[nearest_steuerwert_index]['steuerwert']
            if steuerwert_waehrung_index != -1: current_bank_account['steuerwert_waehrung'] = waehrung_list[steuerwert_waehrung_index]['content']
            if nearest_zinsertrag_index != -1: current_bank_account['zinsertrag'] = zinsertrag_list[nearest_zinsertrag_index]['zinsertrag']
            if zinsertrag_waehrung_index != -1: current_bank_account['zinsertrag_waehrung'] = waehrung_list[zinsertrag_waehrung_index]['content']
            if nearest_zinsertrag_vstpfl_index != -1: current_bank_account['zinsertrag_vstpfl'] = zinsertrag_vstpfl_list[nearest_zinsertrag_vstpfl_index]['zinsertrag_vstpfl']
            if zinsertrag_vstpfl_waehrung_index != -1: current_bank_account['zinsertrag_vstpfl_waehrung'] = waehrung_list[zinsertrag_vstpfl_waehrung_index]['content']
            if nearest_zinsaufwand_index != -1: current_bank_account['zinsaufwand'] = zinsaufwand_list[nearest_zinsaufwand_index]['zinsaufwand']
            if zinsaufwand_waehrung_index != -1: current_bank_account['zinsaufwand_waehrung'] = waehrung_list[zinsaufwand_waehrung_index]['content']
            if nearest_spesen_index != -1: current_bank_account['spesen'] = spesen_list[nearest_spesen_index]['spesen']
            if spesen_waehrung_index != -1: current_bank_account['spesen_waehrung'] = waehrung_list[spesen_waehrung_index]['content']
            if nearest_vst_index != -1: current_bank_account['vst'] = vst_list[nearest_vst_index]['vst']
            if vst_waehrung_index != -1: current_bank_account['vst_waehrung'] = waehrung_list[vst_waehrung_index]['content']
            if nearest_abzug_index != -1: current_bank_account['abzug'] = abzug_list[nearest_abzug_index]['abzug']
            if abzug_waehrung_index != -1: current_bank_account['abzug_waehrung'] = waehrung_list[abzug_waehrung_index]['content']

            bank_account_list.append(current_bank_account)

    return bank_account_list

def returnNumberOrNoneFromString(number_str):
    number = None
    try:   
        number = float(number_str)
    except (ValueError, TypeError) as e:
        logger.error(e)

    return number


while True:
    #Query the database for documents that need to be understood.
    logger.info('Another round!')
    query = 'SELECT [Id], [FileBinary] FROM [KSTADiverses_Test].[dbo].[vdu_bank_statements] WHERE ProcessingStatus = 1'
    connKstaDiversesTest = pyodbc.connect('driver={%s};server=%s;port=1433;database=%s;uid=%s;pwd=%s;TDS_Version=8.0;Encrypt=yes;TrustServerCertificate=YES' % ( config.DRIVER, config.SERVER, config.DB, config.USER, config.PASSWORD ))
    #logger.info("Driver: {} - Server: {} - DB: {} - User: {}".format( config.DRIVER, config.SERVER, config.DB, config.USER))
    cursor = connKstaDiversesTest.cursor()
    cursor.execute(query)
    for row in cursor.fetchall():
        logger.info(row)

        document_id = int(row[0])
        image = Image.open(BytesIO(row[1]))
        image = image.convert("RGB")

        boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='deu')

        resultText = ''
        count = 0
        for i in range(len(boxes['level'])):
            if boxes['text'][i].strip() != '':
                rel_left = boxes['left'][i]/image.size[0]
                rel_top = boxes['top'][i]/image.size[1]
                rel_width = boxes['width'][i]/image.size[0]
                rel_height = boxes['height'][i]/image.size[1]
                conf = boxes['conf'][i] #confidence -1 | 0-100
                block_num = boxes['block_num'][i] #box number
                par_num = boxes['par_num'][i] #paragraph number
                line_num = boxes['line_num'][i] #line number
                word_num = boxes['word_num'][i] #word number
                level = boxes['level'][i] #level

                word = {'id':count,
                        'text':boxes['text'][i].strip(),
                        'left':rel_left,
                        'top':rel_top,
                        'width':rel_width,
                        'height':rel_height,
                        'conf':conf,
                        'block_num':block_num,
                        'par_num':par_num,
                        'line_num':line_num,
                        'word_num':word_num,
                        'level':level
                        }
                #if word['conf'] > 20:
                resultText += (f'{json.dumps(word)},')
                count = count + 1

        resultText = '{"labels":[' + resultText.rstrip(',') + ']}'

        ocr = json.loads(resultText)

        words = []
        bboxes = []

        for label in ocr['labels']:
            words.append(label['text'])
            x1 = round(float(label["left"]) * 1000)
            y1 = round(float(label["top"]) * 1000)
            x2 = round((float(label["left"]) * 1000) + (float(label["width"]) * 1000))
            y2 = round((float(label["top"]) * 1000) + (float(label["height"]) * 1000))
            bbox = [x1,y1,x2,y2]
            bboxes.append(bbox)

        print (words)
        print (bboxes)

        ner_tag_dummies = [0] * len(bboxes)

        encoded_inputs = processor(image, words, boxes=bboxes, word_labels=ner_tag_dummies, padding="max_length", truncation=True, return_tensors="pt")

        #display(image)
        logger.info('image:')
        logger.info(image)
        predictions, token_boxes, image = getPred(encoded_inputs)
        logger.info('inputs:')
        logger.info(encoded_inputs)
        logger.info('predictions:')
        logger.info(predictions)
        logger.info('token_boxes:')
        logger.info(token_boxes)
        bankaccount_list = []
        bankaccount_list = get_bank_account_list()
        logger.info(bankaccount_list)
        if len(bankaccount_list) > 0:
            for bankaccount in bankaccount_list:
                insert_query = 'INSERT INTO [KSTADiverses_Test].[dbo].[vdu_bank_statement_details] (CreationDateTime, IBAN, Startdatum, Abschlussdatum, Zinsertrag, ZinsertragWaehrung, ZinsertragVstPfl, ZinsertragVstPflWaehrung, Verrechnungssteuer, VerrechnungssteuerWaehrung, Spesen, SpesenWaehrung, Steuerwert, SteuerwertWaehrung, BankStatementId) VALUES (CONVERT(datetime,?), ?, CONVERT(datetime,?), CONVERT(datetime,?), CONVERT(decimal(9,2),?), ?, CONVERT(decimal(9,2),?), ?, CONVERT(decimal(9,2),?), ?, CONVERT(decimal(9,2),?), ?, CONVERT(decimal(19,2),?), ?, ?)'
                iban = bankaccount['iban'] if 'iban' in bankaccount.keys() else None
                
                startdatum = bankaccount['startdatum'] if 'startdatum' in bankaccount.keys() else None
                try:
                    startdatum = datetime.datetime.strptime(startdatum, '%d.%m.%y')
                    str_startdatum = startdatum.strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    try: 
                        startdatum = datetime.datetime.strptime(startdatum, '%d.%m.%Y')
                        str_startdatum = startdatum.strftime("%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        str_startdatum = None
                logger.info('startdatum:')
                logger.info(str_startdatum)

                abschlussdatum = bankaccount['abschlussdatum'] if 'abschlussdatum' in bankaccount.keys() else None
                try:
                    abschlussdatum = datetime.datetime.strptime(abschlussdatum, '%d.%m.%y')
                    str_abschlussdatum = abschlussdatum.strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    try: 
                        abschlussdatum = datetime.datetime.strptime(abschlussdatum, '%d.%m.%Y')
                        str_abschlussdatum = abschlussdatum.strftime("%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        str_abschlussdatum = None
                logger.info('abschlussdatum:')
                logger.info(abschlussdatum)

                zinsertrag = returnNumberOrNoneFromString(bankaccount['zinsertrag'] if 'zinsertrag' in bankaccount.keys() else None)
                zinsertrag_waehrung = bankaccount['zinsertrag_waehrung'] if 'zinsertrag_waehrung' in bankaccount.keys() else None
                zinsertrag_vstpfl = returnNumberOrNoneFromString(bankaccount['zinsertrag_vstpfl'] if 'zinsertrag_vstpfl' in bankaccount.keys() else None)
                zinsertrag_vstpfl_waehrung = bankaccount['zinsertrag_vstpfl_waehrung'] if 'zinsertrag_vstpfl_waehrung' in bankaccount.keys() else None
                vst = returnNumberOrNoneFromString(bankaccount['vst'] if 'vst' in bankaccount.keys() else None)
                vst_waehrung = bankaccount['vst_waehrung'] if 'vst_waehrung' in bankaccount.keys() else None
                spesen = returnNumberOrNoneFromString(bankaccount['spesen'] if 'spesen' in bankaccount.keys() else None)
                spesen_waehrung = bankaccount['spesen_waehrung'] if 'spesen_waehrung' in bankaccount.keys() else None
                steuerwert = returnNumberOrNoneFromString(bankaccount['steuerwert'] if 'steuerwert' in bankaccount.keys() else None)
                steuerwert_waehrung = bankaccount['steuerwert_waehrung'] if 'steuerwert_waehrung' in bankaccount.keys() else None
                str_datetime_now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                logger.info('datetime_now:')
                logger.info(str_datetime_now)
                params = (str_datetime_now, iban, str_startdatum, str_abschlussdatum, zinsertrag, zinsertrag_waehrung, zinsertrag_vstpfl, zinsertrag_vstpfl_waehrung, vst, vst_waehrung, spesen, spesen_waehrung, steuerwert, steuerwert_waehrung, document_id)

                connKstaDiversesTest_insert = pyodbc.connect('driver={%s};server=%s;port=1433;database=%s;uid=%s;pwd=%s;TDS_Version=8.0;Encrypt=yes;TrustServerCertificate=YES' % ( config.DRIVER, config.SERVER, config.DB, config.USER, config.PASSWORD ))

                insert_cursor = connKstaDiversesTest_insert.cursor()
                queryResult = insert_cursor.execute(insert_query, params)
                insert_cursor.commit()
                insert_cursor.close()

        update_query = 'UPDATE [KSTADiverses_Test].[dbo].[vdu_bank_statements] SET [ProcessingStatus] = 3 WHERE [Id] = {}'.format(document_id)

        connKstaDiversesTest_update = pyodbc.connect('driver={%s};server=%s;port=1433;database=%s;uid=%s;pwd=%s;TDS_Version=8.0;Encrypt=yes;TrustServerCertificate=YES' % ( config.DRIVER, config.SERVER, config.DB, config.USER, config.PASSWORD ))

        update_cursor = connKstaDiversesTest_update.cursor()
        queryResult = update_cursor.execute(update_query)
        update_cursor.commit()
        update_cursor.close()

    cursor.close()
    time.sleep(10)
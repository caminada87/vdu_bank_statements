import logging, pathlib, time, pyodbc
from os import path
from importlib.machinery import SourceFileLoader

import pytesseract, json, torch, re
from transformers import LayoutLMv2ForTokenClassification
from transformers import LayoutLMv2Processor
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('VDU_bank_statements')

config_path = path.join(pathlib.Path().resolve(), 'instance', 'config.py')
logger.info(f"Config path: {config_path}")
config_module= SourceFileLoader("config",config_path).load_module()
config = config_module.Config()

label_list = ['other', 'iban_key', 'iban', 'abschlussdatum_key', 'abschlussdatum', 'zinsertrag_key', 'zinsertrag', 'steuerwert_key', 'steuerwert', 'waehrung_key', 'waehrung', 'startdatum_key', 'startdatum', 'zinsaufwand_key', 'zinsaufwand', 'vst_key', 'vst', 'spesen_key', 'spesen', 'zinsertrag_vstpfl_key', 'zinsertrag_vstpfl', 'abzug_key', 'abzug']
id2label = {v: k for v, k in enumerate(label_list)}
label2id = {k: v for v, k in enumerate(label_list)}

model = LayoutLMv2ForTokenClassification.from_pretrained("./model", num_labels=len(label_list))
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def get_plaintext_boxcontents_and_coordinates(label, predictions, token_boxes, encoded_inputs):
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

def get_iban_list(predictions, token_boxes, encoded_inputs):
    plaintext_boxcontents = get_plaintext_boxcontents_and_coordinates('iban', predictions, token_boxes, encoded_inputs)

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

def get_number_list(label, predictions, token_boxes, encoded_inputs):
    if label not in list(id2label.values()):
        return []
    plaintext_boxcontents = get_plaintext_boxcontents_and_coordinates(label, predictions, token_boxes, encoded_inputs)
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

def get_date_list(label, predictions, token_boxes, encoded_inputs):
    if label not in list(id2label.values()):
        return []
    plaintext_boxcontents = get_plaintext_boxcontents_and_coordinates(label, predictions, token_boxes, encoded_inputs)
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



while True:
    #Query the database for documents that need to be understood.
    query = 'SELECT [FileBinary] FROM [KSTADiverses_Test].[dbo].[vdu_bank_statements] WHERE ProcessingStatus = 1'
    connKstaDiversesTest = pyodbc.connect('driver={%s};server=%s;database=%s;uid=%s;pwd=%s;Encrypt=yes;TrustServerCertificate=YES' % ( config.DRIVER, config.SERVER, config.DB, config.USER, config.PASSWORD ))
    cursor = connKstaDiversesTest.cursor()
    cursor.execute(query)
    for row in cursor.fetchall():
        logger.info(row)
        break
    cursor.close()
    time.sleep(60)
import csv
import json

with open('Chatbotqa3_qwen25_br_tw_500.csv', mode='r', encoding='utf-8') as csv_file, open('question.jsonl', mode='w', encoding='utf-8') as jsonl_file:
    csv_reader = csv.DictReader(csv_file)
    serial_number = 161
    for row in csv_reader:
        row['serial_number'] = serial_number
        row['category'] = "roleplay"
        row['truns'] = [row['question']]
        del row['question']
        # 写入到 JSONL 文件
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        # 序号加1
        serial_number += 1
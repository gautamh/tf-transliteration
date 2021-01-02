import csv
import re

CAPTURE_REGEX = r'(al[-_]*)|([a-zA-Z0-9-_]*[0-9-_])?([a-zA-Z0-9-_]+[a-zA-Z])([0-9-_]?[a-zA-Z0-9-_]*)?'
USERNAME_INPUT_FILEPATH = r'C:\Users\Gautam\Downloads\TwitterAr2EnTranslit\Username-translation-top50k.txt'
USERNAME_OUTPUT_FILEPATH = r'C:\Users\Gautam\OneDrive\projects\tf-transliteration\cleaned_usernames.txt'
#USERNAME_INPUT_FILEPATH = '/mnt/c/Users/Gautam/Downloads/TwitterAr2EnTranslit/Username-translation-short.txt'

cleaned_names = []

print('Processing input names/usernames...')
with open(USERNAME_INPUT_FILEPATH, 'r', encoding='utf8') as infile:
    reader = csv.DictReader(infile, delimiter='\t')
    for row in reader:
        extracted_username = re.sub(CAPTURE_REGEX, r'\1\3', str(row['UserScreenName']))
        cleaned_username = re.sub(r'[-_]', '', extracted_username).lower()
        cleaned_names.append([row['NormUserName'], row['UserScreenName'], cleaned_username])

print('Writing cleaned data...')
with open(USERNAME_OUTPUT_FILEPATH, 'w+', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerows(cleaned_names)

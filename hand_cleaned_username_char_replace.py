import csv

USERNAME_INPUT_FILEPATH = r'C:\Users\Gautam\OneDrive\projects\tf-transliteration\hand_cleaned_usernames.txt'
USERNAME_OUTPUT_FILEPATH = r'C:\Users\Gautam\OneDrive\projects\tf-transliteration\processed_usernames.txt'

cleaned_pairs = []
HA = 'ه'
TAA_MARBUTA = 'ة'

YA = 'ي'
ALIF_MAQSURAH = 'ى'


print('Processing input names/usernames...')
rows_processed = 0
with open(USERNAME_INPUT_FILEPATH, 'r', encoding='utf8', errors='ignore') as infile:
    reader = csv.reader(infile, delimiter='\t')
    for row in reader:
        if rows_processed % 1000 == 0:
            print('Processed {} rows'.format(rows_processed))
        arabic_uncorr = row[0].strip()
        arabic_corr = arabic_uncorr
        english = row[1].strip()
        if arabic_uncorr.endswith(HA) and (english.endswith('a') or english.endswith('ah')):
            arabic_corr = arabic_uncorr[:-1] + TAA_MARBUTA
        elif arabic_uncorr.endswith(YA) and (english.endswith('a') or english.endswith('ah')):
            arabic_corr = arabic_uncorr[:-1] + ALIF_MAQSURAH
        cleaned_pairs.append([arabic_corr, english])
        rows_processed = rows_processed + 1

print('Writing cleaned data...')
with open(USERNAME_OUTPUT_FILEPATH, 'w+', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerows(cleaned_pairs)
        

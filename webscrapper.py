import requests
from bs4 import BeautifulSoup
import json
import csv
# smiles list to store the smiles of the compounds from the chembl website
smiles = []
for i in range (1 , 10 ) :
#iterating through the chembl website to get the smiles of the compounds
    url = "https://ebi.ac.uk/chembl/compound_report_card/CHEMBL" + str(i)
    content = requests.get(url)
    soup = BeautifulSoup(content.text, 'lxml')
# getting the smiles of the compounds from the script tag. 
# Note: the smiles are in the script tag with the type attribute as application/ld+json and not in the table tag.
    table = soup.find('script', attrs= {'type':'application/ld+json'})
# some compounds are deleted from chembl dataset, thus corresponding chembl ids are empty.
# not all chembl ids have compounds. So, we need to check if the chembl id is empty or not.
# if empty, we skip that chembl id and continue with the next one.
    if table is None:
        continue 
# data present in form of json, so we need to convert it to python dictionary. 
    data = json.loads(table.text)
    print(str(data['smiles'][0]))
# restoring the url to get the smiles of the next compound
    url = "https://ebi.ac.uk/chembl/compound_report_card/CHEMBL"
    a = { "smiles" : str(data['smiles'][0])}
# appending the smiles of the compounds to the smiles list    
    smiles.append(a)

# writing the smiles of the compounds to the csv file
with open ( "smiles.csv" , mode = "w") as smiles_file:
    fieldnames = smiles[ 0 ].keys()
    writer = csv.DictWriter(smiles_file, fieldnames=fieldnames)
    for row in smiles:
        writer.writerow(row)
smiles_file.close()    


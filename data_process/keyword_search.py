import re


Env_keywords = [
    'adaptation ', 'agricultural ', 'air quality ', 'animal ', 'atmospher ', 'biodiversity ', ' biomass ', 'capture ', 'ch4 ', 'climat ', 'co2 ', 'coastal ', 
    'concentration ', 'conservation ', 'consumption ', ' degree ', 'depletion ', 'dioxide ', 'diversity ', 'drought ', 'ecolog ', 'ecosystem ', 'ecosystems ', 
    'emission ', ' emissions ', 'energy ', 'environment ', 'environmental ', ' flood ', 'footprint ', 'forest ', 'fossil ', 'fuel ', 'fuels ', 'gas ', 'gases ', 
    'ghg ', 'global warming ', 'green ', ' greenhouse ', 'hydrogen ', 'impacts ', 'land use ', ' methane ', 'mitigation ', 'n2o ', 'nature ', 'nitrogen ', 
    ' ocean ', 'ozone ', 'plant ', 'pollution ', 'rainfall ', ' renewable ', 'resource ', 'seasonal ', 'sediment ', 'snow ', 'soil ', 'solar ', 'sources ', 
    'sustainab ', 'temperature ', 'thermal ', 'trees ', 'tropical ', 'waste ', 'water', 'recycling', 'clean energy', 'natural'
]

Soc_keywords = [
    'age ', 'cultur ', 'rac ', 'access to ', ' accessibility ', 'accident ', 'accountability ', ' awareness ', 'behaviour ', 'charit ', 'civil ', 
    'code of conduct ', 'communit ', 'community ', 'consumer protection ', 'cyber security ', 'data privacy ', 'data protection ', 'data security ', 
    'demographic ', 'disability ', 'disable ', 'discrimination ', 'divers ', 'donation ', 'education ', 'emotion ', 'employee benefit ', 'employee development ',
     'employment benefit ', 'empower ', 'equal ', 'esg ', ' ethic ', 'ethnic ', 'fairness ', 'family ', 'female ', ' financial protectio ', 'gap ', 'gender ', 
     'health ', 'human ', 'inclus ', 'information security ', 'injury ', 'leave ', 'lgbt ', 'mental well -being ', 'parity ', 'pay equity ', 'peace ', 
     'pension benefit ', 'philanthrop ', 'poverty ', 'privacy ', 'product quality ', 'product safety ', ' promotion ', 'quality of life ', 'religion ', 
     'respectful ', 'respecting ', 'retirement benefit ', 'safety ', ' salary ', 'social ', 'society ', 'supply chain transparency ', 'supportive ', 'talent ', 
     'volunteer ', ' wage ', 'welfare ', 'well -being ', 'wellbeing ', 'wellness ', 'women ', 'workforce ', 'working conditions ', 'product liability',  
]

Gov_keywords = [
    'audit ', 'authority ', 'practice ', ' bribery ', 'code ', 'compensation ', 'competition ', ' competitive ', 'compliance ', 'conflict of interest ', 
    ' control ', 'corporate ', 'corruption ', 'crisis ', 'culture ', 'decision ', 'due diligence ', 'duty ', 'ethic ', ' governance ', 'framework ', 'issue ', 
    'structure ', ' guideline ', 'integrity ', 'internal ', 'lead ', 'legal ', ' lobby ', 'oversight ', 'policy ', 'politic ', 'procedure ', 'regulat ', 
    'reporting ', 'responsib ', 'right ', ' management ', 'sanction ', 'stake ', 'standard ', ' transparen ', ' vot ', 'whistleblower ', 'accounting ', 
    ' accountable ', 'accountant ', 'accounted ', 'board diversity', 'corporate ethics', 'executive pay', 'shareholder rights', 'sustainable governance', 
    'ethical business practices', 'corporate transparency', 'risk management', 'stakeholder engagement', 'anti-corruption', 'business', 'shareholder', 'ownership',
   'board of directors ', 'firm', 'controversy'
]


# 将关键词编译为正则表达式
env_pattern = re.compile('|'.join(Env_keywords), re.IGNORECASE)
soc_pattern = re.compile('|'.join(Soc_keywords), re.IGNORECASE)
gov_pattern = re.compile('|'.join(Gov_keywords), re.IGNORECASE)



def classify_text(input_file, env_output, soc_output, gov_output, non_esg_output):
    with open(input_file, 'r') as file, \
         open(env_output, 'w') as env_file, \
         open(soc_output, 'w') as soc_file, \
         open(gov_output, 'w') as gov_file, \
         open(non_esg_output, 'w') as non_esg_file:

        for line in file:
            if env_pattern.search(line):
                env_file.write(line)
            elif soc_pattern.search(line):
                soc_file.write(line)
            elif gov_pattern.search(line):
                gov_file.write(line)
            else:
                non_esg_file.write(line)


input_file_path = 'merged_output.txt'
classify_text(input_file_path, 'env.txt', 'soc.txt', 'gov.txt', 'non_esg.txt')


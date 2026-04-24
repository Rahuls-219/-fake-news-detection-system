"""
generate_dataset.py
--------------------
Generates a synthetic but realistic training dataset for the Fake News Detection System.
Run this ONCE to create data/news_dataset.csv if you don't have a real dataset.
"""

import pandas as pd
import random

random.seed(42)

FAKE_NEWS = [
    "BREAKING: Scientists discover miracle cure for all diseases hidden by Big Pharma for decades",
    "SHOCKING: Government secretly putting mind control chemicals in drinking water, whistleblower reveals",
    "You won't believe what they found inside the moon — NASA is covering this up!",
    "ALERT: 5G towers are being used to spread the virus, experts are being silenced",
    "Celebrity confirms: vaccines contain microchips to track your every move",
    "EXPOSED: The moon landing was faked in a Hollywood studio, new documents prove",
    "World leaders secretly meet to plan global population reduction agenda, leaked documents show",
    "URGENT: New law will allow government to seize all private property starting next month",
    "Doctors HATE this one weird trick that cures cancer in 24 hours",
    "BREAKING: Aliens have landed in Nevada, military is hiding the truth from the public",
    "SHARE BEFORE DELETED: The truth about fluoride in water will shock you",
    "Government paid crisis actors staged the shooting, family members speak out",
    "This banned video shows the real cause of climate change — it's not CO2!",
    "Scientists admit they lied about evolution for 150 years, real story suppressed",
    "EXPOSED: Major banks are planning to eliminate cash and control your spending",
    "Miracle fruit cures diabetes overnight — pharmaceutical companies don't want you to know",
    "SHOCKING TRUTH: The earth is actually flat and space agencies are all lying",
    "Celebrity whistleblower reveals Hollywood elite runs secret satanic rituals",
    "New report shows COVID was engineered in a lab as a bioweapon by world powers",
    "ALERT: Social media companies listening to your private conversations 24/7",
    "The food pyramid is a lie designed to make you sick and dependent on medicine",
    "BREAKING: Giant ancient civilization discovered underground, media blackout imposed",
    "Scientist fired after proving that chemtrails are real and contain toxic chemicals",
    "Your smartphone is recording everything you say and sending it to the government",
    "EXPOSED: Famous actor faked his death and is living in a secret location",
    "Hidden cancer cure suppressed by drug companies for the last 50 years finally revealed",
    "URGENT WARNING: Common household items being used to spy on families by tech giants",
    "Ancient prophecy predicts the end of the world will happen this year, signs are clear",
    "BREAKING: Major earthquake deliberately triggered by secret military weapon system",
    "They are putting estrogen in plastic to control the population growth, leaked memo confirms",
    "SHARE IMMEDIATELY: New tax will charge citizens for breathing outdoor air next year",
    "Hollywood star reveals the dark truth about the entertainment industry's secret agenda",
    "Scientists confirm that the sun is actually getting colder and ice age is coming soon",
    "EXPOSED: All major elections for the past 20 years have been rigged by elites",
    "BREAKING: Entire city disappears overnight, government covering up the incident",
    "New study proves that chocolate cures all forms of depression better than medication",
    "ALERT: New Internet law will allow government to read all private messages starting Monday",
    "Famous billionaire secretly funding the new world order takeover, sources reveal",
    "SHOCKING: Major airline company caught putting drugs in cabin air to control passengers",
    "Scientists discover ancient giant humans that the history books have been hiding from us",
    "BREAKING: Cure for HIV found in common plant, suppressed by pharmaceutical companies",
    "Government planning to replace police with robots to control citizens, insider says",
    "EXPOSED: Famous charity organization secretly funneling money to terrorist groups",
    "URGENT: Banks planning to close all accounts and switch to digital currency by month end",
    "New evidence proves that dinosaurs never existed — fossils are all fake, researcher says",
    "BREAKING: Secret underground tunnels connect major world capitals for elite travel",
    "They are putting birth control in fast food to reduce world population, whistleblower reveals",
    "SHOCK CLAIM: Major tech CEO admits AI is already smarter than humans and hiding it",
    "ALERT: New radiation from 5G towers causing mass bird deaths across the country",
    "BREAKING: Ancient aliens built the pyramids, Egyptian government finally admits truth",
]

REAL_NEWS = [
    "Federal Reserve raises interest rates by 25 basis points amid ongoing inflation concerns",
    "NASA's James Webb Space Telescope captures detailed images of distant galaxies",
    "World Health Organization releases updated guidelines on antibiotic resistance prevention",
    "Global renewable energy capacity increased by 50 percent over the past decade, report finds",
    "Scientists publish new research linking sedentary lifestyle to increased cardiovascular risk",
    "International trade negotiations between US and EU resume after six-month pause",
    "City council approves new public transportation budget for infrastructure upgrades",
    "Study finds Mediterranean diet associated with lower risk of heart disease in older adults",
    "Tech company announces quarterly earnings exceeding analyst expectations by 12 percent",
    "New climate report urges governments to accelerate carbon emission reduction targets",
    "Supreme Court hears arguments in landmark case involving digital privacy rights",
    "University researchers develop new battery technology that charges 40 percent faster",
    "Local school district launches new STEM program to improve student outcomes",
    "Stock markets close higher after positive jobs report released by labor department",
    "Scientists sequence genome of rare endangered species to assist conservation efforts",
    "Health officials confirm seasonal flu vaccine is 60 percent effective this year",
    "New legislation introduced to address growing concerns about data privacy online",
    "Astronomers detect gravitational waves from merging neutron stars 130 million light years away",
    "Global summit on biodiversity concludes with agreement to protect 30 percent of land by 2030",
    "Electric vehicle sales surpass diesel car sales for the first time in European market",
    "Researchers find promising new drug compound effective against antibiotic-resistant bacteria",
    "Central bank releases quarterly economic outlook projecting moderate growth ahead",
    "New archaeological findings shed light on ancient trade routes in the Mediterranean region",
    "Hospital system partners with university to expand rural telehealth services statewide",
    "Scientists confirm ozone layer recovery is on track following international treaty",
    "Record voter turnout reported in local elections across several major cities",
    "New bridge construction project to begin after years of planning and environmental review",
    "Study shows early childhood education programs reduce long-term inequality in outcomes",
    "Government announces new funding for affordable housing construction in urban areas",
    "Tech giant faces antitrust investigation from European regulators over market practices",
    "Researchers achieve milestone in quantum computing reaching new error correction threshold",
    "Public health campaign successfully reduces smoking rates to historic lows nationally",
    "International aid organizations deliver relief supplies to flood-affected regions",
    "Scientists discover new species of deep-sea fish near hydrothermal vents in Pacific Ocean",
    "City announces expansion of bike lane network to reduce traffic congestion downtown",
    "Annual report shows significant decline in workplace injuries following new safety regulations",
    "Pharmaceutical company begins Phase 3 clinical trials for promising Alzheimer's treatment",
    "New trade agreement expected to increase agricultural exports by 15 percent over five years",
    "Urban planners propose green corridor connecting parks across metropolitan area",
    "University study examines impact of social media usage on adolescent mental health outcomes",
    "Central government releases updated budget with increased spending on infrastructure",
    "Scientists report Antarctic ice sheet is melting faster than previously predicted models showed",
    "New public library opens offering expanded digital resources and community programs",
    "Researchers map brain activity patterns associated with decision-making and reward processing",
    "International aviation body updates safety protocols following review of recent incidents",
    "Local government introduces composting program to reduce municipal solid waste by 30 percent",
    "Economic indicators show steady recovery in manufacturing sector over past two quarters",
    "Scientists develop new water purification method effective against emerging contaminants",
    "Global health agency reports progress in eliminating polio in previously endemic regions",
    "New regulations require food companies to clearly label allergens on all packaged products",
]

# Build dataset
records = []
for text in FAKE_NEWS:
    records.append({"text": text, "label": "FAKE"})
for text in REAL_NEWS:
    records.append({"text": text, "label": "REAL"})

# Augment with slight variations to increase dataset size
augmented = []
for rec in records:
    augmented.append(rec)
    words = rec["text"].split()
    if len(words) > 6:
        # Add a variation with shuffled middle words
        mid = words[2:-2]
        random.shuffle(mid)
        variation = " ".join(words[:2] + mid + words[-2:])
        augmented.append({"text": variation, "label": rec["label"]})

df = pd.DataFrame(augmented)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/news_dataset.csv", index=False)
print(f"Dataset created: {len(df)} samples ({len(df[df.label=='FAKE'])} fake, {len(df[df.label=='REAL'])} real)")

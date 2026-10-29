"""Gold evaluation set. Each question lists title substrings; a retrieved chunk is correct
if its citation title contains ANY of them. `oos` questions must be refused (abstain)."""
GOLD = [
 ("What is the difference between a food allergy and a food intolerance?", ["intolerances", "food allergies"]),
 ("What is IgE and what does it do?", ["What are food allergies"]),
 ("Why do some people get food allergies, why me?", ["Why me"]),
 ("How does a skin prick test work?", ["Tests › Skin Prick", "Tests"]),
 ("What is a food challenge?", ["Food Challenges", "Tests"]),
 ("What are co-factors that can make a reaction worse?", ["Co-factors"]),
 ("Can I drink alcohol if I have a food allergy?", ["Alcohol"]),
 ("Do I need to tell the airline about my allergy when I fly?", ["Travel"]),
 ("What are the signs of anaphylaxis?", ["Anaphylaxis"]),
 ("What should I do if someone is having anaphylaxis?", ["Anaphylaxis", "Adrenaline", "Steps"]),
 ("How does adrenaline work?", ["How does adrenaline work"]),
 ("How do I use my adrenaline pen step by step?", ["Steps"]),
 ("How should I store my adrenaline pen?", ["How do I store adrenaline", "Steps"]),
 ("What is the difference between adrenaline and epinephrine?", ["adrenaline and epinephrine", "Steps"]),
 ("What is Neffy nasal spray?", ["Neffy"]),
 ("Is asthma a risk factor with food allergy?", ["Asthma"]),
 ("Which antihistamine can I take for a mild reaction?", ["Antihistamines"]),
 ("Is coconut a tree nut?", ["Tree nuts"]),
 ("Is a peanut actually a nut?", ["Peanut"]),
 ("What foods often contain peanuts?", ["Peanut"]),
 ("Where is sesame hidden in food?", ["Sesame"]),
 ("Is couscous or semolina wheat?", ["Wheat"]),
 ("Which foods are shellfish crustaceans?", ["Crustaceans"]),
 ("Are mussels and oysters molluscs?", ["Molluscs"]),
 ("Is soy sauce made from soya?", ["Soya"]),
 ("If I am allergic to one tree nut can I try other tree nuts?", ["What if I'm allergic to some tree nuts"]),
 ("Can I get a reaction from kissing someone who ate my allergen?", ["Behavioural › Behavioural", "kissing", "Can you get a reaction from kissing"]),
 ("Can I get anaphylaxis from touching or smelling my food allergen?", ["touching or smelling", "Alcohol"]),
 ("How do I read food labels for allergens?", ["Avoidance"]),
 ("What are the top 14 allergens?", ["Allergens", "Avoidance"]),
 ("Which foods contain egg?", ["Egg"]),
 ("Is fish sauce a problem for fish allergy?", ["Fish"]),
 ("What foods have milk in them?", ["Milk"]),
 ("What allergens are on the training menu satay chicken skewers?", ["Satay"]),
 ("Does the prawn tempura contain gluten?", ["Prawn Tempura"]),
]
# Out-of-scope: must be refused.
OOS = [
 "What is the capital of France?", "How do I bake a chocolate cake?", "Who won the football match yesterday?",
 "Explain how to write a Python function", "What is the weather today?", "Tell me a joke about cats",
 "How do I fix my bicycle tyre?", "What is the best phone to buy?",
]
# In-scope phrasing that must NOT be refused.
IN_SCOPE = [
 "Can I kiss my boyfriend if he ate peanuts?", "My friends keep pushing me to try cake, what do I say?",
 "do I need an epipen at school", "what is anaphylaxis", "can I eat out at restaurants safely",
 "what happens if I accidentally eat my allergen", "what is a peanut allergy",
]

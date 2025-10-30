how to use
  
(Main)		(code place)		               	(text place)	   	(words)		            (file type)		        (count of needed words)
python "D:\klasifikator_enchanted.py" --dir "D:\texty" --keywords "auto,car,engine" --extensions .txt --use-fuzzy --min-score 6

(can be use)
--fuzzy-threshold 80


python "D:\klasifikator.py" --dir "D:\texty\txt" --keywords "cm3, HP, motor, vykon, kilowatt, kW, hmotnost, vaha, kg, dlzka, sirka, vyska, rozmery, rozmer, objem, liter, l, nafta, benzin, diesel, EURO 3, EURO 6, EURO 7, spotreba, hybrid, nadrz, koleso, pneumatika, dvere, pocet dveri, prevodovka, pohon, brzda, kapacita, nosnost, sedadlo, miesto, pasazier, rychlost, km/h, motorovy, valec, cylindrov, farba, emisny, CO2" --extensions .txt --use-fuzzy --min-score 10 --out "D:\texty\results.csv"

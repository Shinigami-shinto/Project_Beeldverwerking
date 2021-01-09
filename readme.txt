Om het project te runnen voer je main.py uit met als argument: -s "pad naar de foto" 

Om de databank opnieuw aan te maken unzip je database.zip naar een map, en voer je generateJson.py uit met als argument: -s "pad naar de databankfolder"
Er wordt dan een database.bin file aangemaakt die gebruikt wordt in de andere scripts

Om de testlabels aan te maken (dictionary van filenaam en welke zaal deze file is) unzip je testSetPerZaal.zip en voer je generateJson_test.py uit met als argument: -s "pad naar folder 
met test afbeeldingen per zaal"
Er wordt dan een testlabels.bin file aangemaakt die gebruikt wordt in de andere scripts 

Om het project te evalueren voer je evaluate_testset.py uit met als argument: -s "pad naar map met test afbeeldingen per zaal"
Dit script gebruikt de testlabels om te controlleren of de predictie juist is, dus moet je zeker de testlabels.bin hebben aangemaakt
zoals hierboven beschreven.

voorbeeld om te evalueren:
Eerst unzip je testSetPerZaal.zip naar /home/user/testset/
/home/user/testset/ bevat nu per zaal een map met afbeeldingen
voer de volgende commando's uit:
python generateJson_test.py -s "/home/user/testset/"
python evaluate_testset.py -s "/home/user/testset/"

voorbeeld om main te runnen:
Als database.bin al bestaat moet je gewoon het volgende commando uitvoeren:
python main.py "/home/user/mapMetAfbeelding/afbeelding.jpg"

Om DEMO.PY te runnen verander je "path_vids" in de code naar de map waar alle video's in zitten. Dan run je het script met
met als enige argument de volledige filenaam van de video

voorbeeld om demo te runnen:
Je hebt een map /home/users/vids waar een video in zit met als naam MSK_12.mp4
Verander path_vids naar "/home/users/vids" en voer volgend commando uit:
python DEMO.py MSK_12.mp4

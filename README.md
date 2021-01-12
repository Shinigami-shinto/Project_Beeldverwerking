# Computer Vision Project: Waar bevind ik mij in het MSK van Gent? 
## Inleiding
<img src="./doel_project.PNG" alt="hi"/>

## In het verslag geschreven in LaTeX staat al de nodige informatie: Computer_Visie__Paintings_MSK.pdf

## Korte toelichting voor het uitvoeren van de code
Om het project te runnen voer je main.py uit met als argument: -s "pad naar de foto" 

voorbeeld om main te runnen:
Als database.bin al bestaat moet je gewoon het volgende commando uitvoeren:
python main.py "/home/user/mapMetAfbeelding/afbeelding.jpg"

Het programma gaat dan na waar je je bevindt in het MSK aan de hand van de foto die je met je smartphone hebt genomen van een schilderij afbeelding.jpg

Bij de Demo wordt het plaatsbepalingsalgoritme toegpast op een video waar een gebruiker door het museum wandeld, telkens een schilderij in beeld komt, wordt de beste match weergegeven en de zaal waarin je je bevindt word weergegeven.

Om DEMO.PY te runnen verander je "path_vids" in de code naar de map waar alle video's in zitten. Dan run je het script met
met als enige argument de volledige filenaam van de video

voorbeeld om demo te runnen:
Je hebt een map /home/users/vids waar een video in zit met als naam MSK_12.mp4
Verander path_vids naar "/home/users/vids" en voer volgend commando uit:
python DEMO.py MSK_12.mp4

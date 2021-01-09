2 mappen:
	textureGenerator:
		dient puur voor het aanmaken van 'patches' (of kleine textuur afbeeldingen)
		textureMaker.py runnen:
			-s met het pad naar de dataset (hier overloopt hij alle bestanden) en toont ze een voor een random
			-d doel waar je de patches wilt opslaan
			(deze worden nog eens onderverdeeld in een submap met de klasse die je gekozen hebt)

			wanneer je een afbeelding te zien krijgt kan je een letter kiezen (ik heb 'a' of 'z' gekozen) deze worden dan gebruikt als label van je klasse.
			'a' => 'niet schilderij'
			'z' => 'wel schilderij'
			wanneer je een afbeelding ziet en je drukt op a dan zal het programma een patch toevoegen van 100x100px waar je muis zich bevindt.
			druk je bijvoorbeeld op 'a' wanneer je over de muur hovert zal een patch toegevoegd worden in doelMap/a/a-randomNummer.jpg

	textureClassifier:
		recognize.py gebruikt een SVM om de patches dan te classificeren. (1/3 is fout)
		recognizeNN.py gebruikt een Neuraal Net (3/62) is fout.

		beide besanden gebruiken LBP's en gemiddelde kleurwaarden om de textures te beschrijven.

		recognizeNN.py runnen:
		-t trainingsset => waarbij de afbeeldingen in een map zitten die de naam van de klasse heeft, bijvoorbeeld de map patches in textureGenerator
		-e evaluatie map, map met afbeeldingen die je wilt testen.

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.fftpack import dct, idct

slika = Image.open("../../img/ursa_major_1.jpg")

stupci, retci = slika.size
print("Broj stupaca: " + str(stupci))
print("Broj redaka: " + str(retci))
if (slika.mode == "RGB"):
    print("Broj kanala: 3")
else:
    print("Broj kanala: 1")

slika_grayscale = slika.convert('L')
slika_matrix = np.asarray(slika_grayscale)

velicina_bloka = 40

broj_horizontalnih_blokova = stupci // velicina_bloka
broj_vertikalnih_blokova = retci // velicina_bloka

broj_trenutnog_bloka = 1

for i in range(0, broj_vertikalnih_blokova):
    for j in range(0, broj_horizontalnih_blokova):
        draw = ImageDraw.Draw(slika)
        draw.rectangle(((j*velicina_bloka, i*velicina_bloka),
                        (j*velicina_bloka+velicina_bloka, i*velicina_bloka+velicina_bloka)))
        broj_trenutnog_bloka_1 = str(broj_trenutnog_bloka)
        draw.text((j*velicina_bloka, i*velicina_bloka), broj_trenutnog_bloka_1)
        broj_trenutnog_bloka = broj_trenutnog_bloka + 1

slika.show()
slika.save("blokovi.png")

trenutni_pocetni_stupac = 0
trenutni_pocetni_redak = -velicina_bloka

frekvencije = [0] * (broj_horizontalnih_blokova * broj_vertikalnih_blokova)
frekvencije2 = [0] * (broj_horizontalnih_blokova * broj_vertikalnih_blokova)
frekvencije3 = [0] * (broj_horizontalnih_blokova * broj_vertikalnih_blokova)
brojac_frekvencija = 0

for redak in range(broj_vertikalnih_blokova):
    trenutni_pocetni_stupac = 0
    trenutni_pocetni_redak = trenutni_pocetni_redak + velicina_bloka

    for stupac in range(broj_horizontalnih_blokova):
        temp_matrix = np.zeros(shape=(velicina_bloka, velicina_bloka))
        temp_matrix_dct = np.zeros(shape=(velicina_bloka, velicina_bloka))
        trenutni_redak = 0

        for i in range(trenutni_pocetni_redak, trenutni_pocetni_redak + velicina_bloka):
            trenutni_stupac = 0
            trenutni_redak = 0

            for j in range(trenutni_pocetni_stupac, trenutni_pocetni_stupac + velicina_bloka):
                temp_matrix[trenutni_redak,
                            trenutni_stupac] = slika_matrix[i, j]
                trenutni_stupac = trenutni_stupac + 1
                trenutni_redak = trenutni_redak + 1

        print("Trenutni blok: ")
        print(temp_matrix)
        print("\n")

        temp_matrix_dct = dct(dct(temp_matrix.T).T)

        print("DCT matrica za trenutni blok:")
        print(temp_matrix_dct)
        print("\n")

        frekvencije[brojac_frekvencija] = temp_matrix_dct[0, 0]
        frekvencije2[brojac_frekvencija] = temp_matrix_dct[1, 1]
        frekvencije3[brojac_frekvencija] = temp_matrix_dct[2, 2]
        brojac_frekvencija = brojac_frekvencija + 1

        trenutni_pocetni_stupac = trenutni_pocetni_stupac + velicina_bloka

for i in range(0, broj_horizontalnih_blokova * broj_vertikalnih_blokova):
    print("Blok " + str(i+1) + ": (1) " + str(frekvencije[i]) + " (2) " + str(
        frekvencije2[i]) + " (3) " + str(frekvencije3[i]))

print("***Sada se ispisuje medusobna slicnost blokova...***")
for i in range(0, broj_horizontalnih_blokova * broj_vertikalnih_blokova):
    print("Blok " + str(i+1) + ": (1) " + str(frekvencije[i]) + " (2) " + str(
        frekvencije2[i]) + " (3) " + str(frekvencije3[i]))
    for j in range(i+1, broj_horizontalnih_blokova * broj_vertikalnih_blokova):
        if((abs(frekvencije[i] - frekvencije[j]) < 10) and (abs(frekvencije2[i] - frekvencije2[j]) < 10) and (abs(frekvencije3[i] - frekvencije3[j]) < 10)):
            print("---> Slican je sljedecim blokovima: Blok " + str(j+1) + ": (1) " +
                  str(frekvencije[j]) + " (2) " + str(frekvencije2[j]) + " (3) " + str(frekvencije3[j]))

print("ZAVRŠETAK!\n")
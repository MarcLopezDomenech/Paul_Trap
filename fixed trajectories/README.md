# Nova versió de la geometria i potencial del electròdes

Diria bastant convençut que ara ja va bé el càlcul de les trajectòries.

## Canvis
* <b>Forma de la geometria</b>. He canviat un poc la forma dels electròdes, de manera que ara estan més curvats i el radi de l'hiperboloide d'una fulla és significativament menor (de 5m l'he reduït a 2m).
* <b>Precisió de la geometria</b>. Abans hi havia 1300 triangles en total i ara 3768.
* <b>Precisió del potencial en tot l'espai</b>. Primer, he calculat V.npy que està construida en una meshgrid 51 x 51 x 51 i crec que ja funcionava bé. Recomeno utilitzar V2.npy que està construida en una meshgrid de 101 x 101 x 101. Amb V2.npy, ion_trajectory.py i trajectory_freq.py van a una velocitat decent.
* <b>Simulació</b>. Algun canvi en el plot animat per adaptar-lo a un nombre arbitrari de partícules i he afegit l'opció de generar posicions i velocitats inicials aleatòries.

Si voleu executar electrodes_potential.py amb una meshgrid de 101 x 101 x 101 (el que hi ha posat ara), tingueu en compte que a mi m'ha tardat 36 min en calcular V2.npy.

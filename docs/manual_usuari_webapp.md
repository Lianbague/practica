# Manual d'Usuari — Aplicació Web de Conversió de Formats de Dataset

**Projecte:** Gestió i Transformació de Formats de Dataset per a Deep Learning
**Autor:** Lian
**Versió:** 1.0
**Data:** Juny 2026

---

## Descripció

L'aplicació web permet convertir datasets d'imatges entre els quatre formats de dataset més utilitzats en Deep Learning:

- **ImageFolder** — estructura de carpetes, una per classe
- **HDF5** — fitxer binari únic (.h5)
- **NPZ** — arxiu comprimit de NumPy (.npz)
- **TFRecord** — format natiu de TensorFlow (.tfrecord)

Totes les conversions inclouen una **validació automàtica** que comprova que cada píxel s'ha preservat exactament durant la conversió.

---

## Accés a l'aplicació

L'aplicació és accessible públicament des de qualsevol navegador web, sense necessitat d'instal·lar res:

> **URL:** `https://practica-[id].streamlit.app`

No cal registre ni compte. Obriu l'enllaç i l'aplicació es carrega directament.

---

## Com utilitzar l'aplicació

L'aplicació es divideix en tres passos seqüencials.

---

### Pas 1 — Carregar el dataset d'origen

Teniu dues opcions per proporcionar el dataset:

#### Opció A: Pujar un fitxer (recomanada per a accés remot)

| Format | Com pujar-lo |
|--------|-------------|
| HDF5 | Arrossegueu el fitxer `.h5` directament |
| NPZ | Arrossegueu el fitxer `.npz` directament |
| TFRecord | Comprimiu el fitxer `.tfrecord` i el `.tfrecord.meta.json` junts en un ZIP i pugeu el ZIP |
| ImageFolder | Comprimiu la carpeta del dataset (amb les subcarpetes de classe) en un ZIP i pugeu el ZIP |

Feu clic a **"Inspect dataset"** per confirmar que el dataset s'ha carregat correctament. L'aplicació mostrarà:
- Nombre de mostres
- Nombre de classes
- Mida de les imatges
- Una graella de 6 imatges de mostra

> **Nota:** Si el dataset ImageFolder té més de 10 classes, apareixerà un control per limitar el nombre de classes a convertir.

#### Opció B: Introduir una ruta (només en execució local)

Si executeu l'aplicació al vostre propi ordinador, podeu introduir directament la ruta del fitxer o carpeta, per exemple:
```
/Users/lian/datasets/train.h5
/Users/lian/tiny-imagenet-200/train
```

---

### Pas 2 — Seleccionar els formats de destinació

Un cop inspeccionat el dataset, apareix el Pas 2.

- Seleccioneu un o més formats de destinació mitjançant les caselles de selecció.
- El format d'origen no apareix com a opció (no té sentit convertir al mateix format).
- Feu clic al botó **"Convert"** per iniciar la conversió.

La conversió pot trigar entre uns pocs segons i un parell de minuts, depenent de la mida del dataset.

---

### Pas 3 — Descarregar i verificar els resultats

Un cop acabada la conversió, apareix el Pas 3 amb una targeta per cada format convertit. Cada targeta mostra:

- **Nom del format i icona**
- **Mida del fitxer** resultant en MB
- **Estat de la validació** (✅ correcte / ❌ error)
- **Botó de descàrrega** del fitxer convertit

#### Detalls de validació

Feu clic a qualsevol targeta per expandir els detalls de validació, que inclou:

1. **Nombre de mostres** — coincideix amb l'original
2. **Forma de la imatge** — coincideix amb l'original
3. **Tipus de dades** — coincideix amb l'original
4. **Rang de valors** — coincideix amb l'original
5. **Igualtat d'etiquetes** — les etiquetes de classe no han canviat
6. **Distribució de classes** — la proporció per classe és idèntica
7. **MSE de píxels** — l'error quadràtic mitjà és inferior a 1×10⁻⁶ (pràcticament zero)

A sota dels checks de validació trobareu una **comparació visual** d'imatges originals i convertides, una al costat de l'altra, per confirmar que les imatges són idèntiques visualment.

---

## Preguntes freqüents

**El fitxer és massa gran per pujar-lo?**
El límit de pujada és de 500 MB. Per a datasets més grans, utilitzeu l'aplicació en local (vegeu el Manual Tècnic).

**La conversió ha fallat. Què faig?**
Expandiu la targeta del format que ha fallat. Hi trobareu el missatge d'error i el rastre complet per diagnosticar el problema.

**Puc convertir entre qualsevol parell de formats?**
Sí. Qualsevol dels 4 formats pot ser origen i qualsevol dels altres 3 pot ser destinació. Hi ha 12 parells dirigits possibles, tots suportats.

**Per a ImageFolder, la validació mostra una advertència ⚠️ en lloc de ✅. És un problema?**
No. Quan el destí és ImageFolder, l'ordre de càrrega canvia perquè les imatges es reorganitzen per carpetes de classe. Les dades (píxels i etiquetes) estan completament preservades; l'advertència indica que l'ordre de les mostres és diferent, la qual cosa és el comportament esperat.

---

## Requisits tècnics

- Qualsevol navegador web modern (Chrome, Firefox, Safari, Edge)
- Connexió a internet
- Cap instal·lació necessària

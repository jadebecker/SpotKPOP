# SpotKPOP
Use Spotify API to predict wether as song is a Kpop song or not

# Built with
* [Keras with TensorFlow](https://keras.io/) Use to build model
* [Spotipy](http://spotipy.readthedocs.io/en/latest/) Use to access Spotify API
* [Numpy](http://www.numpy.org/) Use to build model
* [Sklearn](http://scikit-learn.org/stable/index.html) Use to build model
* [H5py](https://www.h5py.org/) Use to save model

# Running classifier

```
python3 spotclass.py username
```

# Running predictor

```
python3 PredictorSpot.py username
```

# Output example
```
Killing Me Softly with His Song - Fugees
PAS KPOP 

Up & Down - EXID
KPOP

DDD - EXID
KPOP
```
PAS KPOP = Not a Kpop song / KPOP = Kpop song

# Problem

The model can recognize Kpop songs but also recognizes pop songs as kpop songs.

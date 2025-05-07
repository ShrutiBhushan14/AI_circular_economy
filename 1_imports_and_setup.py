
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import folium
from folium.plugins import HeatMap
from datetime import datetime
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. SETUP PARAMETERS
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
NUM_CLASSES = 6 

# 2. DATA AUGMENTATION & DATA LOADERS
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for testing
)

train_generator = datagen.flow_from_directory(
    'dataset/train', 
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Added Validation Generator (Crucial for Conference Results)
validation_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 

# 3. BUILD THE MODEL
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze base layers
base_model.trainable = False

# Add custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x) # Reduced to 512 for faster training
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. COMPILE AND TRAIN
# Using a slightly lower learning rate is better for Transfer Learning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# 

print("🚀 Starting Training...")
# Store history to plot graphs later for your project report
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# 5. SAVE THE MODEL
model.save('tomato_inception_v3.keras')
print("✅ Model saved as tomato_inception_v3.keras")
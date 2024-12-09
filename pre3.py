from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Define the target size for the images
target_size = (256, 256)

# Resize function
def resize_images(images, target_size):
    resized_images = []
    for img in images:
        img = Image.fromarray((img * 255).astype('uint8'))
        img = img.resize(target_size)
        resized_images.append(np.array(img) / 255.0)
    return np.array(resized_images)

sar_images_resized = resize_images(sar_images, target_size)
optical_images_resized = resize_images(optical_images, target_size)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)

# Apply augmentation
augmented_sar_images = datagen.flow(sar_images_resized, batch_size=32, shuffle=True)
augmented_optical_images = datagen.flow(optical_images_resized, batch_size=32, shuffle=True)
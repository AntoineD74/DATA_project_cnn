from model.py import ModelResnet

def train():
    """*********************** Import data********************"""
    
    images, labels, labels2=common.read_json('training.json', 20)
    images=np.array(images, dtype=np.float32)/255
    labels=np.array(labels, dtype=np.float32)
    index=np.random.permutation(len(images))
    images=images[index]
    labels=labels[index]
    
    
    """********************* Prepare data *********************"""
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    test_dataset = test_dataset.batch(batch_size)
    
    
    """******************* Train our model ********************"""
    
    num_classes = 10  # Replace with the actual number of classes in your dataset
    model = ModelResnet(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Step 4: Train the model
    num_epochs = 10  # Adjust the number of epochs as needed
    batch_size = 32  # Adjust the batch size as needed
    model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)
    
    # Step 5: Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

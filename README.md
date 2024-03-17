# Medical-Handwritten-Prescription-Recognition
Author - Subrata Maity
Methodology--->
1.	Data Preprocessing:
•	This code snippet prepares the training data for character-based machine learning tasks by analyzing the labels to find the maximum sequence length and the size of the unique character vocabulary. This information can be useful for tasks like building character-level language models or text recognition models.
•	It decodes the image as a grayscale image (with one channel) using tf.image.decode_png(image, 1). You might need to modify this line depending on your image format (e.g., use channels=3 for RGB images).
•	It resizes the image using the distortion_free_resize function (defined previously) to ensure aspect ratio preservation.
•	It normalizes the pixel values by dividing by 255.0 (assuming pixel values are in the 0-255 range) and casting them to float32 for better performance in machine learning models.
•	It uses char_to_num (defined previously) to convert the label text (split into characters) into a sequence of integer IDs representing each character.
•	It calculates the amount of padding needed (pad_amount) to make the sequence the same length as the maximum sequence length (max_len) identified earlier.
•	It uses tf.pad to pad the label sequence with the padding_token at the end to reach the desired length.
•	It applies the process_images_labels function in parallel using map with num_parallel_calls=AUTOTUNE to potentially improve efficiency.
•	It caches the batched dataset using .cache() to potentially improve training speed by reusing preprocessed data.
•	It prefetches data using .prefetch(AUTOTUNE) to potentially overlap data preprocessing with model training for better performance.
These functions work together to preprocess image and label data, converting images to a consistent size and format, encoding text labels into integer sequences, and preparing batches of data for training a character-level machine learning model in TensorFlow.

2.	Model Architecture:
•	Two convolutional blocks with ReLU activation and max pooling for feature extraction:
	Conv1: 32 filters, (3x3) kernel size, He Normal initialization, "same" padding.
	Conv2: 64 filters, (3x3) kernel size, He Normal initialization, "same" padding.
•	Reshapes output into a sequence format for RNN processing, accounting for downsampling in pooling layers.
•	Dense layer with 64 units and ReLU activation, followed by dropout regularization (20%) for dimensionality reduction.
•	Two bidirectional LSTM layers with dropout (25%) for capturing long-range dependencies in both directions:
	First LSTM: 128 units, returning sequences for subsequent processing.
	Second LSTM: 64 units, returning sequences.
•	Dense layer with (number of characters + 2) units and softmax activation, predicting a probability distribution over characters for each time step. The "+2" accounts for CTC blank and CTC-specific token.
•	Custom layer calculating Connectionist Temporal Classification (CTC) loss, well-suited for tasks where input and output sequences have different lengths and alignment is unknown (e.g., optical character recognition).
•	Designed for optical character recognition (OCR) or similar tasks involving text recognition from images.
•	Combines convolutional layers for feature extraction with recurrent layers for sequence modeling.
•	Employs CTC loss for alignment-free training and prediction.

3.	Training and Evaluation:
•	The model.compile line now explicitly sets loss=CTCLayer() to ensure the correct loss function is used during training.
•	It casts the labels batch to a sparse tensor using tf.sparse.from_dense.
•	It uses keras.backend.ctc_decode to decode the CTC predictions into character sequences (only considering the first greedy decoding path and keeping at most max_len characters).
•	The decoded sequences are cast to a sparse tensor using tf.sparse.from_dense.
•	It uses tf.edit_distance to calculate the edit distance (Levenshtein distance) between each pair of corresponding sparse tensors (labels and predictions).
•	It computes the mean edit distance across the entire batch using tf.reduce_mean.
•	EditDistanceCallback class custom callback inherits from keras.callbacks.Callback to monitor and report edit distance during training.
•	These functions provide a way to evaluate the model's performance based on edit distance (number of edits required to transform predicted text into ground truth text) during training. The callback monitors this metric on the validation set and reports it after each epoch.
•	This approach uses greedy decoding for predictions, which might not always be the optimal path. You could explore beam search decoding for potentially better results.
•	Edit distance is a good metric for evaluating character-level errors, but it might not capture all aspects of performance, like semantic meaning. Consider additional evaluation metrics depending on your specific needs.

4.	Post-processing and Reporting:
•	The code demonstrates an effective approach for extracting medicine information from image predictions generated by a character recognition model.
•	By incorporating the suggested improvements, you can further enhance its robustness and accuracy.

5.	Data Handling and Processing:
•	This code processes image data and corresponding labels, sorts them potentially based on filename order, and creates a text file with the information organized in a table format for further use.
•	It defines a custom sorting function (get_numeric_part) to sort the image paths based on any numeric part within the filenames (assuming filenames have numbers indicating order).
•	Finally, it saves the DataFrame as a tab-delimited text file (output_image.txt) at the specified location.
•	This code cleans and prepares a text dataset for further use by removing unnecessary lines and randomly shuffling the remaining entries for unbiased processing or analysis.
•	It skips lines that start with "#" (likely comments or metadata).
•	It checks if the first word in a line is "err" and skips those lines (considered as errored entries).
•	It imports the numpy library (aliased as np) and calls np.random.shuffle to randomly shuffle the order of the elements in the words_list.
•	This code facilitates a common practice in machine learning by dividing a dataset into training, validation, and testing sets, which serve different purposes in model development and evaluation.
•	It uses an assert statement to ensure that the total number of samples in the original list equals the sum of the samples in the three sets, verifying that no data was lost during splitting.
•	This code processes the previously split training, validation, and testing data sets to extract the actual image paths and their corresponding labels, potentially for use in training and evaluating an image recognition model.
•	The code calls the get_image_paths_and_labels function three times: Once with train_samples to get training image paths and labels, once with validation_samples to get validation image paths and labels, and once with test_samples to get testing image paths and labels.

6.	Visualization:
•	It uses plt.subplots from Matplotlib to create a figure with a 5x5 grid of subplots for displaying 25 images.
•	It sets the figure size to (15, 8) for better visualization.
•	It flips the image horizontally using tf.image.flip_left_right.
•	It uses tf.gather and tf.where to filter out padding tokens from the label sequence. This ensures only actual characters are considered for display.
•	It uses tf.strings.reduce_join to combine the filtered character IDs back into a single string representation.
•	It uses ax[i // 5, i % 5].imshow to display 25 images.

By organizing the report, a clear and structured overview of the code's functionality is provided data preprocessing, model architecture, training process, evaluation metrics, and potential improvements.



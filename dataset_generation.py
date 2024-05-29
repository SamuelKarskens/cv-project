from music21 import note, stream, converter
import cv2

pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']  # Example pitches
output_folder = 'dataset2'

for pitch in pitches:
    n = note.Note(pitch)
    s = stream.Stream([n])
    s.write('lily.png', fp=f'{output_folder}/{pitch}', subformat=['png'], templatePath='template.ly')

    image = cv2.imread(f'{output_folder}/{pitch}.png')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get a binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert the binary image
    binary = cv2.bitwise_not(binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Save the cropped image
    output_path = f'{output_folder}/{pitch}_cropped.png'
    cv2.imwrite(output_path, cropped_image)

    print(f'Cropped image saved to {output_path}')


import AsposeOcr 

recognitionEngine = AsposeOcr()
# Add image to batch
input = OcrInput(InputType.SINGLE_IMAGE)
input.add("sample.png")
# Extract text from image
result = recognitionEngine.recognize(input)
# Display the recognition result
print(result[0].recognition_text)
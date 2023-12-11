from mainModel import Balooning_CSV_Generator

# Take user input for the PDF path
pdf_path = input("Enter the PDF path: ")

# Take user input for the API choice
APIChoice = input("Do you want to use API (Yes/No): ")

print(Balooning_CSV_Generator(pdf_path, APIChoice))

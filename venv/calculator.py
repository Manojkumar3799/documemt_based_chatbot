import sys

print("Arguments received:", sys.argv)
# Check if two numbers were provided
if len(sys.argv) != 3:
    print("Usage: python calculator.py <num1> <num2>")
    sys.exit()

# Get the two numbers from command line arguments (as strings)
num1_str = sys.argv[1]
num2_str = sys.argv[2]

# Convert them to numbers (floats)
num1 = float(num1_str)
num2 = float(num2_str)

# Add the numbers
result = num1 + num2

# Print the result
print("Result:", result)

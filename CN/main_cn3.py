def generate_hamming(data):
    # Convert input string to list of integers
    d = [int(x) for x in data]

    # Set the Parity Bits
    p1 = d[0] ^ d[1] ^ d[3]
    p2 = d[0] ^ d[2] ^ d[3]
    p4 = d[1] ^ d[2] ^ d[3]

    # Build the 7-box package: [P1, P2, D1, P4, D2, D3, D4]
    return [p1, p2, d[0], p4, d[1], d[2], d[3]]


def detect_and_correct(received):
    # Convert input string to list of integers
    r = [int(x) for x in received]

    # Run the Receiver's Checks
    c1 = r[0] ^ r[2] ^ r[4] ^ r[6]
    c2 = r[1] ^ r[2] ^ r[5] ^ r[6]
    c4 = r[3] ^ r[4] ^ r[5] ^ r[6]

    # Calculate error position (binary to decimal)
    error_pos = (c4 * 4) + (c2 * 2) + (c1 * 1)

    if error_pos == 0:
        print("\nStatus: PASS - No error detected.")
        return r
    else:
        print(f"\nStatus: FAIL - Error detected at Box {error_pos}!")

        # Fix the error by flipping the broken bit (0 to 1, or 1 to 0)
        r[error_pos - 1] ^= 1
        print(f"Status: Box {error_pos} successfully corrected.")
        return r


# --- MAIN PROGRAM ---
print("--- SENDER ---")
# Ask user for the original 4-bit message
data_input = input("Enter exactly 4 bits of data (e.g., 1010): ")

# Generate and display the 7-bit package
encoded_data = generate_hamming(data_input)
encoded_str = "".join(str(x) for x in encoded_data)
print(f"Transmitted 7-bit package: {encoded_str}")

print("\n--- RECEIVER ---")
# Ask user to type the received package
prompt_msg = (
    f"Enter the 7 bits received (try changing one bit from {encoded_str}): "
)
received_input = input(prompt_msg)

# Detect and fix any errors
corrected_data = detect_and_correct(received_input)
corrected_str = "".join(str(x) for x in corrected_data)
print(f"Final corrected package: {corrected_str}")
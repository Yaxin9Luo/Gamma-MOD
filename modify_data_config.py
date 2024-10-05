import json

input_file_path = '/data/luogen_code/MGM/playground/data/mgm_instruction.json'
output_file_path = '/data/luogen_code/MGM/playground/data/modified_mgm_instruction.json'
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Print one or two examples from the data to check the structure
print("First example:")
print(json.dumps(data[0], indent=4))

if len(data) > 1:
    print("\nSecond example:")
    print(json.dumps(data[1], indent=4))
    print(json.dumps(data[2], indent=4))
    print(json.dumps(data[3], indent=4))
    print(json.dumps(data[4], indent=4))
    print(json.dumps(data[5], indent=4))

# with open(input_file_path, 'r') as file:
#     data = json.load(file)

# for item in data:
#     if 'conversations' in item:
#         for convo in item['conversations']:
#             if 'value' in convo:
#                 value = convo['value']
#                 if value.endswith('\n<image>'):
#                     # Remove '\n<image>' from the end
#                     value = value[:-8]
#                     # Add '<image>\n' at the beginning
#                     value = '<image>\n' + value
#                     # Update the value in the item
#                     convo['value'] = value

# with open(output_file_path, 'w') as file:
#     json.dump(data, file, indent=4)

# print(f'Modified data has been saved to {output_file_path}')

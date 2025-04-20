#!/usr/bin/env python3
import json
import sys
import os
import re
import argparse
from datetime import datetime
import signal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate Verilog test cases using a local language model')
    parser.add_argument('--input', required=True, help='Input JSON file with module I/O information')
    parser.add_argument('--output', help='Output JSON file for test cases (default: <input_base>_testcases.json)')
    parser.add_argument('--no-model', action='store_true', help='Skip LLM and use template-based generation')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout in seconds for model generation (default: 120)')
    parser.add_argument('--model', default='Salesforce/codegen-350M-mono', help='Model to use for generation (default: Salesforce/codegen-350M-mono)')
    return parser.parse_args()

def load_io_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def initialize_model(model_name):
    """Initialize a language model for test case generation"""
    print(f"Initializing language model: {model_name}...")
    try:
        # Check for available hardware acceleration
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA acceleration")
        else:
            device = "cpu"
            print("Using CPU for computation")
        
        # Create pipeline with better configuration for reliable output
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Make sure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,  # Use half precision on GPU/MPS
            low_cpu_mem_usage=True,
            device_map=device
        )
        
        return model, tokenizer
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Falling back to smaller model...")
        try:
            # Fallback to a smaller but still code-capable model
            fallback_model = "Salesforce/codegen-125M-mono"
            print(f"Trying fallback model: {fallback_model}")
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                torch_dtype=torch.float32,
                device_map="auto" 
            )
            return model, tokenizer
        except Exception as e:
            print(f"Error initializing fallback model: {e}")
            return None, None

def timeout_handler(signum, frame):
    raise TimeoutError("Generation is taking too long")

def generate_test_cases_with_model(model, tokenizer, module_info, timeout=120):
    # Format the module info to create a clear prompt
    input_list = '\n'.join([f"- {port['name']} (width: {port['width']})" for port in module_info['inputs']])
    output_list = '\n'.join([f"- {port['name']} (width: {port['width']})" for port in module_info['outputs']])
    inout_list = '\n'.join([f"- {port['name']} (width: {port['width']})" for port in module_info['inouts']])
    
    # Build the prompt for the model - more structured for better output
    prompt = f"""Generate Verilog test cases for the following module:

Module name: {module_info['module_name']}

Input ports:
{input_list}

Output ports:
{output_list}

{"Inout ports:" if module_info['inouts'] else ""}
{inout_list if module_info['inouts'] else ""}

The test cases should be in the following JSON format:
```json
{{
  "test_cases": [
    {{
      "test_id": "tc_001",
      "description": "Specific test case description",
      "inputs": {{
        "input_port_name": "value"
      }},
      "expected_outputs": {{
        "output_port_name": "expected_value"
      }}
    }}
  ]
}}
```

For bit values, use Verilog notation (1'b0, 1'b1). For multi-bit values, use appropriate hex notation (e.g., 8'hFF for 8-bit values).
Create at least 3 different test cases that cover different edge cases and common scenarios.
Ensure the JSON is valid and properly formatted.

JSON output:
```json
"""

    # Generate text with timeout
    print("Generating test cases with language model...")
    print(f"Timeout set to {timeout} seconds")
    
    # Set up timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        # Tokenize with explicit attention mask
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(inputs.input_ids)
        
        # Generate with more focused parameters for structured output
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,      # More tokens to ensure complete output
                do_sample=False,          # Turn off sampling for more deterministic output
                num_beams=1,              # Simple beam search
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask  # Explicit attention mask
            )
        
        # Decode the generated text
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Cancel alarm
        signal.alarm(0)
        
        # Print some debug info
        print("Successfully generated output from model")
        
        return full_text
    except TimeoutError:
        print("Model generation timed out")
        signal.alarm(0)  # Ensure alarm is canceled
        return None
    except Exception as e:
        print(f"Error during generation: {e}")
        signal.alarm(0)  # Ensure alarm is canceled
        return None

def extract_json_from_response(response_text):
    if response_text is None:
        print("No response to extract JSON from")
        return None
        
    # Extract JSON content from the response with better parsing
    try:
        # Try to find JSON between markers first
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, response_text)
        
        if json_match:
            json_text = json_match.group(1).strip()
            print("Found JSON within code blocks")
        else:
            # Try to find the first { and the last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                print("Extracted JSON block from response")
            else:
                # Try one more approach - find the substring starting with "JSON output:"
                json_output_marker = response_text.find("JSON output:")
                if json_output_marker >= 0:
                    potential_json = response_text[json_output_marker + len("JSON output:"):].strip()
                    start_idx = potential_json.find('{')
                    end_idx = potential_json.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_text = potential_json[start_idx:end_idx]
                        print("Extracted JSON from 'JSON output:' section")
                    else:
                        print("Could not find valid JSON block in the response")
                        return None
                else:
                    print("Could not find JSON block in the response")
                    return None
        
        # Try to parse the extracted JSON
        try:
            parsed_json = json.loads(json_text)
            print("Successfully parsed JSON")
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print("Attempting to fix common JSON formatting issues...")
            
            # Try to fix common issues in the JSON
            # Replace single quotes with double quotes
            fixed_text = json_text.replace("'", '"')
            # Ensure property names have double quotes
            fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
            
            try:
                parsed_json = json.loads(fixed_text)
                print("Successfully parsed JSON after fixing format issues")
                return parsed_json
            except json.JSONDecodeError:
                print("Failed to parse JSON even after attempted fixes")
                print("Problematic JSON content:")
                print(json_text[:200] + "..." if len(json_text) > 200 else json_text)
                return None
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

def generate_template_based_test_cases(module_info):
    """Generate test cases based on templates for common logic modules"""
    print("Generating test cases using template-based approach...")
    
    module_name = module_info['module_name'].lower()
    test_cases = []
    
    # Extract input and output port names
    input_ports = [port['name'] for port in module_info['inputs']]
    output_ports = [port['name'] for port in module_info['outputs']]
    
    # Test case counter
    tc_count = 1
    
    # Basic logic gates templates
    if "and" in module_name:
        # AND gate - create all input combinations
        if len(input_ports) == 2:  # 2-input AND gate
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Both inputs 0",
                "inputs": {input_ports[0]: "1'b0", input_ports[1]: "1'b0"},
                "expected_outputs": {output_ports[0]: "1'b0"}
            })
            tc_count += 1
            
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Input A=0, B=1",
                "inputs": {input_ports[0]: "1'b0", input_ports[1]: "1'b1"},
                "expected_outputs": {output_ports[0]: "1'b0"}
            })
            tc_count += 1
            
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Input A=1, B=0",
                "inputs": {input_ports[0]: "1'b1", input_ports[1]: "1'b0"},
                "expected_outputs": {output_ports[0]: "1'b0"}
            })
            tc_count += 1
            
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Both inputs 1",
                "inputs": {input_ports[0]: "1'b1", input_ports[1]: "1'b1"},
                "expected_outputs": {output_ports[0]: "1'b1"}
            })
        else:  # Multi-input AND gate
            # All zeros
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "All inputs 0",
                "inputs": {port: "1'b0" for port in input_ports},
                "expected_outputs": {output_ports[0]: "1'b0"}
            })
            tc_count += 1
            
            # All ones
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "All inputs 1",
                "inputs": {port: "1'b1" for port in input_ports},
                "expected_outputs": {output_ports[0]: "1'b1"}
            })
            tc_count += 1
            
            # One input at 0, rest at 1
            for i, port in enumerate(input_ports):
                inputs = {p: "1'b1" for p in input_ports}
                inputs[port] = "1'b0"
                test_cases.append({
                    "test_id": f"tc_{tc_count:03d}",
                    "description": f"All inputs 1 except {port}=0",
                    "inputs": inputs,
                    "expected_outputs": {output_ports[0]: "1'b0"}
                })
                tc_count += 1
    
    elif "or" in module_name:
        # OR gate - create all input combinations
        if len(input_ports) == 2:  # 2-input OR gate
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Both inputs 0",
                "inputs": {input_ports[0]: "1'b0", input_ports[1]: "1'b0"},
                "expected_outputs": {output_ports[0]: "1'b0"}
            })
            tc_count += 1
            
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Input A=0, B=1",
                "inputs": {input_ports[0]: "1'b0", input_ports[1]: "1'b1"},
                "expected_outputs": {output_ports[0]: "1'b1"}
            })
            tc_count += 1
            
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Input A=1, B=0",
                "inputs": {input_ports[0]: "1'b1", input_ports[1]: "1'b0"},
                "expected_outputs": {output_ports[0]: "1'b1"}
            })
            tc_count += 1
            
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Both inputs 1",
                "inputs": {input_ports[0]: "1'b1", input_ports[1]: "1'b1"},
                "expected_outputs": {output_ports[0]: "1'b1"}
            })
        else:  # Multi-input OR gate
            # All zeros
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "All inputs 0",
                "inputs": {port: "1'b0" for port in input_ports},
                "expected_outputs": {output_ports[0]: "1'b0"}
            })
            tc_count += 1
            
            # All ones
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "All inputs 1",
                "inputs": {port: "1'b1" for port in input_ports},
                "expected_outputs": {output_ports[0]: "1'b1"}
            })
            tc_count += 1
            
            # One input at 1, rest at 0
            for i, port in enumerate(input_ports):
                inputs = {p: "1'b0" for p in input_ports}
                inputs[port] = "1'b1"
                test_cases.append({
                    "test_id": f"tc_{tc_count:03d}",
                    "description": f"All inputs 0 except {port}=1",
                    "inputs": inputs,
                    "expected_outputs": {output_ports[0]: "1'b1"}
                })
                tc_count += 1
    
    elif "not" in module_name or "inv" in module_name:
        # NOT/Inverter gate
        test_cases.append({
            "test_id": f"tc_{tc_count:03d}",
            "description": "Input 0",
            "inputs": {input_ports[0]: "1'b0"},
            "expected_outputs": {output_ports[0]: "1'b1"}
        })
        tc_count += 1
        
        test_cases.append({
            "test_id": f"tc_{tc_count:03d}",
            "description": "Input 1",
            "inputs": {input_ports[0]: "1'b1"},
            "expected_outputs": {output_ports[0]: "1'b0"}
        })
    
    elif "xor" in module_name:
        # XOR gate
        if len(input_ports) == 2:  # 2-input XOR gate
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Both inputs 0",
                "inputs": {input_ports[0]: "1'b0", input_ports[1]: "1'b0"},
                "expected_outputs": {output_ports[0]: "1'b0"}
            })
            tc_count += 1
            
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Input A=0, B=1",
                "inputs": {input_ports[0]: "1'b0", input_ports[1]: "1'b1"},
                "expected_outputs": {output_ports[0]: "1'b1"}
            })
            tc_count += 1
            
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Input A=1, B=0",
                "inputs": {input_ports[0]: "1'b1", input_ports[1]: "1'b0"},
                "expected_outputs": {output_ports[0]: "1'b1"}
            })
            tc_count += 1
            
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": "Both inputs 1",
                "inputs": {input_ports[0]: "1'b1", input_ports[1]: "1'b1"},
                "expected_outputs": {output_ports[0]: "1'b0"}
            })
    
    # Add register/flip-flop templates
    elif "ff" in module_name or "flop" in module_name or "reg" in module_name:
        # Look for clock and reset inputs
        clock_port = next((p for p in input_ports if any(c in p.lower() for c in ["clk", "clock"])), input_ports[0])
        reset_port = next((p for p in input_ports if any(r in p.lower() for r in ["rst", "reset"])), 
                          input_ports[1] if len(input_ports) > 1 else None)
        
        data_ports = [p for p in input_ports if p != clock_port and p != reset_port]
        
        # Basic register tests
        test_cases.append({
            "test_id": f"tc_{tc_count:03d}",
            "description": "Reset condition",
            "inputs": {
                clock_port: "1'b0",
                **({reset_port: "1'b1"} if reset_port else {}),
                **{dp: "1'b0" for dp in data_ports}
            },
            "expected_outputs": {op: "1'b0" for op in output_ports}
        })
        tc_count += 1
        
        # Data capture on clock edge
        test_cases.append({
            "test_id": f"tc_{tc_count:03d}",
            "description": "Data capture on clock edge",
            "inputs": {
                clock_port: "1'b1",
                **({reset_port: "1'b0"} if reset_port else {}),
                **{dp: "1'b1" for dp in data_ports}
            },
            "expected_outputs": {op: "1'b1" for op in output_ports}
        })
    
    # Add more templates for other common modules as needed
    elif "mux" in module_name:
        # Multiplexer testing
        sel_port = next((p for p in input_ports if any(s in p.lower() for c in ["sel", "select"])), 
                        input_ports[-1])  # Usually the last port is select
        data_ports = [p for p in input_ports if p != sel_port]
        
        # Test each select option
        for i, port in enumerate(data_ports):
            # Create test with only this input active
            inputs = {p: "1'b0" for p in data_ports}
            inputs[port] = "1'b1"
            
            # Set select value - assuming binary select
            if len(data_ports) <= 2:  # 2:1 mux
                inputs[sel_port] = f"1'b{i}"
            else:  # wider mux
                sel_width = (len(data_ports) - 1).bit_length()  # Calculate required select width
                inputs[sel_port] = f"{sel_width}'h{i:x}"  # Hex format
                
            test_cases.append({
                "test_id": f"tc_{tc_count:03d}",
                "description": f"Select input {port} (sel={i})",
                "inputs": inputs,
                "expected_outputs": {output_ports[0]: "1'b1"}
            })
            tc_count += 1
    
    # If no specific template matched, generate generic test cases
    if not test_cases:
        print("No specific template matched. Generating generic test cases.")
        
        # Generate cases with all 0s and all 1s
        test_cases.append({
            "test_id": f"tc_{tc_count:03d}",
            "description": "All inputs 0",
            "inputs": {port: "1'b0" for port in input_ports},
            "expected_outputs": {port: "1'b0" for port in output_ports}
        })
        tc_count += 1
        
        test_cases.append({
            "test_id": f"tc_{tc_count:03d}",
            "description": "All inputs 1",
            "inputs": {port: "1'b1" for port in input_ports},
            "expected_outputs": {port: "1'b1" for port in output_ports}
        })
        tc_count += 1
        
        # For multi-bit ports, use appropriate hex values
        for port in module_info['inputs']:
            if port['width'] > 1:
                width = port['width']
                # Add some test cases with different bit patterns
                test_cases.append({
                    "test_id": f"tc_{tc_count:03d}",
                    "description": f"Test {port['name']} with alternating bits",
                    "inputs": {port['name']: f"{width}'h{'A' * (width//4 + (1 if width % 4 else 0))}"},
                    "expected_outputs": {out_port: "1'b0" for out_port in output_ports}
                })
                tc_count += 1
                
                # Test with all bits set
                test_cases.append({
                    "test_id": f"tc_{tc_count:03d}",
                    "description": f"Test {port['name']} with all bits set",
                    "inputs": {port['name']: f"{width}'h{'F' * (width//4 + (1 if width % 4 else 0))}"},
                    "expected_outputs": {out_port: "1'b1" for out_port in output_ports}
                })
                tc_count += 1
    
    return {"test_cases": test_cases}

def main():
    args = parse_arguments()
    
    # Set default output file if not provided
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_testcases.json"
    
    # Load module I/O data
    module_info = load_io_data(args.input)
    
    print(f"Generating test cases for module: {module_info['module_name']}")
    
    test_cases = None
    
    # Check if we should skip the model generation
    if not args.no_model:
        # Initialize the model
        model, tokenizer = initialize_model(args.model)
        
        if model and tokenizer:
            # Generate test cases with the model
            response_text = generate_test_cases_with_model(model, tokenizer, module_info, args.timeout)
            
            # Extract and parse the JSON from the response
            if response_text:
                test_cases = extract_json_from_response(response_text)
                
                # Print a preview of the response if extraction failed
                if test_cases is None:
                    print("Failed to extract JSON. First 200 chars of response:")
                    print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
    
    # If model generation failed or was skipped, use template-based approach
    if test_cases is None:
        print("Using template-based generation as fallback")
        test_cases = generate_template_based_test_cases(module_info)
    
    # Update test case inputs/outputs based on actual module I/O if needed
    if test_cases:
        # Make sure test cases have valid input/output names
        for tc in test_cases["test_cases"]:
            # Ensure inputs match those in the module definition
            valid_inputs = {port["name"]: tc["inputs"].get(port["name"], "0") 
                           for port in module_info["inputs"]}
            tc["inputs"] = valid_inputs
            
            # Ensure outputs match those in the module definition
            valid_outputs = {port["name"]: tc["expected_outputs"].get(port["name"], "0") 
                            for port in module_info["outputs"]}
            tc["expected_outputs"] = valid_outputs
    
        # Add metadata to the output
        output_data = {
            "module_name": module_info["module_name"],
            "generation_timestamp": datetime.now().isoformat(),
            "test_cases": test_cases["test_cases"]
        }
        
        # Write the test cases to a file
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Generated {len(test_cases['test_cases'])} test cases")
        print(f"Test cases written to: {args.output}")
    else:
        print("Failed to generate test cases")
        sys.exit(1)

if __name__ == "__main__":
    main()

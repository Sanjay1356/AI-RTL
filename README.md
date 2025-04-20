#AI Based RTL Verification
This repository contains a set of tools for automating the testing workflow for Verilog modules. The toolkit includes scripts for parsing Verilog modules, generating test cases, and creating UVM (Universal Verification Methodology) testbenches.
#Overview
The toolkit provides an end-to-end solution for Verilog module verification:

Verilog Parser - Extracts module interfaces (input/output ports) from Verilog files
Test Case Generator - Creates test cases using language models or template-based generation
UVM Generator - Builds complete UVM testbench infrastructure based on the module and test cases
#Scripts
1. Verilog Parser (verilog-parser-script.pl)
This Perl script parses Verilog files to extract module interfaces and saves them in JSON format.
Usage:
./verilog-parser-script.pl <verilog_file>

Features:

Extracts module name, input ports, output ports, and inout ports
Determines port widths
Outputs a JSON file with the format <module_name>_io.json

2. Test Case Generator (gpt4-testcase-generator.py)
This Python script generates test cases for Verilog modules using either a language model or template-based generation.
Usage:
./gpt4-testcase-generator.py --input <module_io.json> [options]
Options:

--input: Input JSON file with module I/O information (required)
--output: Output JSON file for test cases (default: <input_base>_testcases.json)
--no-model: Skip LLM and use template-based generation
--timeout: Timeout in seconds for model generation (default: 120)
--model: Model to use for generation (default: Salesforce/codegen-350M-mono)

Features:

Uses language models to generate smart test cases
Falls back to template-based generation if model is unavailable or times out
Generates test cases for common module types (AND, OR, XOR, MUX, etc.)
Outputs a JSON file with test cases and expected results

3. UVM Generator (uvm-generator-script.pl)
This Perl script generates UVM testbench files from test cases.
Usage:
./uvm-generator-script.pl <testcases_json_file> [output_directory]
Features:

Generates complete UVM environment with all necessary components
Creates test-specific sequences based on test cases
Generates interface definitions, drivers, monitors, and scoreboard
Creates test files for each individual test case and a combined test

#Workflow Example
# 1. Parse the Verilog module
./verilog-parser-script.pl my_module.v
# Output: my_module_io.json

# 2. Generate test cases
./gpt4-testcase-generator.py --input my_module_io.json
# Output: my_module_testcases.json

# 3. Generate UVM testbench
./uvm-generator-script.pl my_module_testcases.json my_uvm_tb/
# Output: Various UVM files in the my_uvm_tb/ directory

Generated UVM Files
The UVM generator creates the following files:

<module_name>_if.sv - Interface
<module_name>_seq_item.sv - Sequence item
<module_name>_sequence.sv - Sequences (including test-specific sequences)
<module_name>_sequencer.sv - Sequencer
<module_name>_driver.sv - Driver
<module_name>_monitor.sv - Monitor
<module_name>_agent.sv - Agent
<module_name>_scoreboard.sv - Scoreboard
<module_name>_env.sv - Environment
<module_name>_test.sv - Test cases
<module_name>_tb_top.sv - Testbench top module

#Requirements
Verilog Parser

Perl
JSON Perl module (cpan install JSON)

Test Case Generator

Python 3
PyTorch
Transformers library (pip install transformers)
(Optional) CUDA or MPS for hardware acceleration

UVM Generator

Perl
JSON Perl module (cpan install JSON)


# AI Based RTL Verification

This repository contains a collection of scripts for automating the generation of Verilog module test infrastructure, including test case generation and UVM testbench creation.

## Overview

The toolkit consists of three main components:

1. **Verilog Parser** - Extracts I/O information from Verilog modules
2. **Test Case Generator** - Creates test cases using either an LLM or template-based approach
3. **UVM Testbench Generator** - Generates complete UVM testbench framework from test cases

These tools work together to automate the manual and time-consuming process of creating testbenches for Verilog modules.

## Components

### 1. Verilog Parser (`verilog-parser-script.pl`)

A Perl script that parses Verilog files to extract module interface information, including:
- Module name
- Input ports (with widths)
- Output ports (with widths) 
- Inout ports (with widths)

This information is saved in a structured JSON format that serves as input for the test case generator.

```bash
# Usage
perl verilog-parser-script.pl <verilog_file>

# Example
perl verilog-parser-script.pl my_module.v
# Outputs: my_module_io.json
```

### 2. Test Case Generator (`gpt4-testcase-generator.py`)

A Python script that generates test cases for Verilog modules using either:
- A language model (LLM) approach
- Template-based generation for common module types

The script creates a set of test cases with input values and expected output values based on the module I/O information.

```bash
# Usage
python gpt4-testcase-generator.py --input <io_json_file> [--output <output_file>] [--no-model] [--timeout <seconds>] [--model <model_name>]

# Example with LLM
python gpt4-testcase-generator.py --input my_module_io.json --timeout 60

# Example with template-based generation (no LLM)
python gpt4-testcase-generator.py --input my_module_io.json --no-model
```

Options:
- `--input`: Input JSON file with module I/O information (required)
- `--output`: Output JSON file for test cases (default: <input_base>_testcases.json)
- `--no-model`: Skip LLM and use template-based generation
- `--timeout`: Timeout in seconds for model generation (default: 120)
- `--model`: Model to use for generation (default: Salesforce/codegen-350M-mono)

### 3. UVM Generator (`uvm-generator-script.pl`)

A Perl script that generates a complete UVM testbench framework from the test cases, including:
- Interface definition
- Sequence items
- Sequences (including test-specific sequences)
- Sequencer
- Driver
- Monitor
- Agent
- Scoreboard
- Environment
- Test classes
- Top-level testbench

```bash
# Usage
perl uvm-generator-script.pl <testcases_json_file> [output_directory]

# Example
perl uvm-generator-script.pl my_module_testcases.json uvm_testbench
```

## Workflow

The typical workflow is:

1. Parse the Verilog module to extract I/O information:
   ```bash
   perl verilog-parser-script.pl my_module.v
   ```

2. Generate test cases:
   ```bash
   python gpt4-testcase-generator.py --input my_module_io.json
   ```

3. Generate UVM testbench:
   ```bash
   perl uvm-generator-script.pl my_module_testcases.json uvm_testbench
   ```

## Dependencies

### Verilog Parser
- Perl
- JSON Perl module

### Test Case Generator
- Python 3.6+
- torch
- transformers
- Language model (optional but recommended)

### UVM Generator
- Perl
- JSON Perl module

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/verilog-uvm-automation.git
   cd verilog-uvm-automation
   ```

2. Install Python dependencies:
   ```bash
   pip install torch transformers
   ```

3. Install Perl dependencies:
   ```bash
   cpan JSON
   ```

4. Ensure scripts are executable:
   ```bash
   chmod +x verilog-parser-script.pl gpt4-testcase-generator.py uvm-generator-script.pl
   ```

## Example

Here's a complete example of using the toolkit with a simple module:

1. Create a Verilog file named `counter.v`:
   ```verilog
   module counter (
       input wire clk,
       input wire rst_n,
       input wire enable,
       output reg [7:0] count
   );
       always @(posedge clk or negedge rst_n) begin
           if (!rst_n)
               count <= 8'h00;
           else if (enable)
               count <= count + 1;
       end
   endmodule
   ```

2. Parse the Verilog file:
   ```bash
   perl verilog-parser-script.pl counter.v
   ```

3. Generate test cases:
   ```bash
   python gpt4-testcase-generator.py --input counter_io.json
   ```

4. Generate UVM testbench:
   ```bash
   perl uvm-generator-script.pl counter_testcases.json counter_uvm
   ```

5. Review the generated UVM files in the `counter_uvm` directory.




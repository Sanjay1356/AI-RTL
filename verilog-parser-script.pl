#!/usr/bin/perl
use strict;
use warnings;
use JSON;

# Input and output file paths
my $verilog_file = $ARGV[0] or die "Usage: $0 <verilog_file>\n";
my $output_json = $verilog_file;
$output_json =~ s/\.\w+$/_io.json/;

# Open the Verilog file
open(my $fh, '<', $verilog_file) or die "Could not open file '$verilog_file': $!";

# Data structures to store module information
my $module_name = "";
my @input_ports = ();
my @output_ports = ();
my @inout_ports = ();
my %port_widths = ();

# Parse the Verilog file
my $in_module = 0;
my $in_port_list = 0;

while (my $line = <$fh>) {
    chomp $line;
    
    # Remove comments
    $line =~ s/\/\/.*$//;
    
    # Skip if line is empty after removing comments
    next if $line =~ /^\s*$/;
    
    # Detect module declaration
    if ($line =~ /^\s*module\s+(\w+)\s*\(/) {
        $module_name = $1;
        $in_module = 1;
        $in_port_list = 1;
        next;
    }
    
    # End of port list
    if ($in_port_list && $line =~ /\);/) {
        $in_port_list = 0;
        next;
    }
    
    # Parse port declarations - input, output and inout
    if ($in_module) {
        if ($line =~ /^\s*input\s+(?:wire|reg)?\s*(?:\[(\d+):(\d+)\])?\s*(\w+)\s*;?/) {
            my ($msb, $lsb, $port) = ($1, $2, $3);
            push @input_ports, $port;
            if (defined $msb && defined $lsb) {
                $port_widths{$port} = abs($msb - $lsb) + 1;
            } else {
                $port_widths{$port} = 1;
            }
        }
        elsif ($line =~ /^\s*output\s+(?:wire|reg)?\s*(?:\[(\d+):(\d+)\])?\s*(\w+)\s*;?/) {
            my ($msb, $lsb, $port) = ($1, $2, $3);
            push @output_ports, $port;
            if (defined $msb && defined $lsb) {
                $port_widths{$port} = abs($msb - $lsb) + 1;
            } else {
                $port_widths{$port} = 1;
            }
        }
        elsif ($line =~ /^\s*inout\s+(?:wire|reg)?\s*(?:\[(\d+):(\d+)\])?\s*(\w+)\s*;?/) {
            my ($msb, $lsb, $port) = ($1, $2, $3);
            push @inout_ports, $port;
            if (defined $msb && defined $lsb) {
                $port_widths{$port} = abs($msb - $lsb) + 1;
            } else {
                $port_widths{$port} = 1;
            }
        }
        
        # Detect the end of the module
        if ($line =~ /^\s*endmodule/) {
            $in_module = 0;
        }
    }
}

close($fh);

# Create a structured data object
my %module_info = (
    module_name => $module_name,
    inputs => [],
    outputs => [],
    inouts => []
);

foreach my $port (@input_ports) {
    push @{$module_info{inputs}}, {
        name => $port,
        width => $port_widths{$port}
    };
}

foreach my $port (@output_ports) {
    push @{$module_info{outputs}}, {
        name => $port,
        width => $port_widths{$port}
    };
}

foreach my $port (@inout_ports) {
    push @{$module_info{inouts}}, {
        name => $port,
        width => $port_widths{$port}
    };
}

# Convert to JSON and write to file
my $json = JSON->new->pretty->encode(\%module_info);
open(my $json_fh, '>', $output_json) or die "Could not open file '$output_json' for writing: $!";
print $json_fh $json;
close($json_fh);

# Print summary
print "Module '$module_name' successfully parsed.\n";
print "Found ", scalar(@input_ports), " input ports, ", scalar(@output_ports), " output ports, and ", scalar(@inout_ports), " inout ports.\n";
print "JSON data written to '$output_json'\n";
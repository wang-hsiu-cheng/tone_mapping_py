//==================================================================================================
//  Note:          SRAM Model for Output (RGB)
//  Based on sram_input.v structure
//==================================================================================================

module sram_output #(     // for output data
    parameter CH_NUM = 3,     // RGB 3 channel (Output is usually 24-bit RGB)
    parameter BW_PER_CH = 8,  // each channel has 8 bits
    parameter HEIGHT = 720,
    parameter WIDTH = 1280,
    parameter ADDR_WIDTH = $clog2(HEIGHT * WIDTH) // log2(921600) = 20
)
(
    input clk,
    input csb,  // chip enable (active low)
    input wsb,  // write enable (active low)
    input [CH_NUM*BW_PER_CH-1:0] wdata, 
    input [ADDR_WIDTH-1:0] waddr, // write address
    input [ADDR_WIDTH-1:0] raddr, // read address

    output reg [CH_NUM*BW_PER_CH-1:0] rdata // read data
);

    reg [CH_NUM*BW_PER_CH-1:0] _rdata;
    // Memory Array
    reg [CH_NUM*BW_PER_CH-1:0] mem [0:HEIGHT*WIDTH-1];

    // Write Operation
    always @(posedge clk) begin
        if(~csb && ~wsb) begin
            mem[waddr] <= wdata;
        end
    end

    // Read Operation
    always @(posedge clk) begin
        if(~csb) begin
            _rdata <= mem[raddr];
        end
    end

    // Output assignment with simulation delay
    always @* begin
        rdata = #(1) _rdata;
    end

    // Task: Backdoor Load (可以預先填入背景色，或用於 Debug)
    task load_act(
        input integer index,
        input [CH_NUM*BW_PER_CH-1:0] param_input
    );
        mem[index] = param_input;
    endtask

    // Task: Reset SRAM (清空輸出緩衝區)
    task reset_sram;
        integer i;
        begin
            for(i=0; i<HEIGHT*WIDTH; i=i+1) begin
                mem[i] = {CH_NUM*BW_PER_CH{1'b0}}; // 預設歸零，方便觀察波形
            end
        end
    endtask

endmodule
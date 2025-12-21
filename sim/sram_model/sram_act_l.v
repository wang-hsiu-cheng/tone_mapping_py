//==================================================================================================
//  Note:          Use only for teaching materials of IC Design Lab, NTHU.
//  Copyright: (c) 2025 Vision Circuits and Systems Lab, NTHU, Taiwan. ALL Rights Reserved.
//==================================================================================================

module sram_act_l #(     //for save log luminance
parameter DATA_WIDTH = 21, // for sign Q7.14
parameter HEIGHT = 720,
parameter WIDTH = 1280,
parameter ADDR_WIDTH = $clog2(HEIGHT * WIDTH) // log2(921600) = 19.81 => 20
)
(
input clk,
input csb,  //chip enable
input wsb,  //write enable
input [DATA_WIDTH-1:0] wdata, 
input [ADDR_WIDTH-1:0] waddr, //write address
input [ADDR_WIDTH-1:0] raddr, //read address

output reg [DATA_WIDTH-1:0] rdata //read data 28 bits
);

reg [DATA_WIDTH-1:0] _rdata;
reg [DATA_WIDTH-1:0] mem [0:HEIGHT*WIDTH-1];

always @(posedge clk) begin
    if(~csb && ~wsb) begin
        mem[waddr] <= wdata;
    end
end

always @(posedge clk) begin
    if(~csb) begin
        _rdata <= mem[raddr];
    end
end

always @* begin
    rdata = #(1) _rdata;
end

task load_act(
    input integer index,
    input [DATA_WIDTH-1:0] param_input
);
    mem[index] = param_input;
endtask

task reset_sram;
    integer i;
    begin
        for(i=0;i<HEIGHT*WIDTH;i=i+1)begin
            mem[i] = {DATA_WIDTH{1'bX}};
        end
    end
endtask

endmodule
module line_buffer (
    input wire clk,
    input wire signed [20:0] log_lum_out,

    output reg signed [20:0] lb00,
    output reg signed [20:0] lb01,
    output reg signed [20:0] lb02,
    output reg signed [20:0] lb03,
    output reg signed [20:0] lb04,
    output reg signed [20:0] lb10,
    output reg signed [20:0] lb11,
    output reg signed [20:0] lb12,
    output reg signed [20:0] lb13,
    output reg signed [20:0] lb14,
    output reg signed [20:0] lb20,
    output reg signed [20:0] lb21,
    output reg signed [20:0] lb22,
    output reg signed [20:0] lb23,
    output reg signed [20:0] lb24,
    output reg signed [20:0] lb30,
    output reg signed [20:0] lb31,
    output reg signed [20:0] lb32,
    output reg signed [20:0] lb33,
    output reg signed [20:0] lb34,
    output reg signed [20:0] lb40,
    output reg signed [20:0] lb41,
    output reg signed [20:0] lb42,
    output reg signed [20:0] lb43,
    output reg signed [20:0] lb44
);

reg signed [20:0] LB0 [0:719];
reg signed [20:0] LB1 [0:719];
reg signed [20:0] LB2 [0:719];
reg signed [20:0] LB3 [0:719];
reg signed [20:0] LB4 [0:4];

integer i;
always @(posedge clk) begin
    // LB4
    for (i = 0; i < 4; i = i + 1) begin
        LB4[i] <= LB4[i+1];
    end
    LB4[4] <= log_lum_out;

    // LB3
    for (i = 0; i < 719; i = i + 1) begin
        LB3[i] <= LB3[i+1];
    end
    LB3[719] <= LB4[0];
    // LB2
    for (i = 0; i < 719; i = i + 1) begin
        LB2[i] <= LB2[i+1];
    end
    LB2[719] <= LB3[0];
    // LB1
    for (i = 0; i < 719; i = i + 1) begin
        LB1[i] <= LB1[i+1];
    end
    LB1[719] <= LB2[0];
    // LB0
    for (i = 0; i < 719; i = i + 1) begin
        LB0[i] <= LB0[i+1];
    end
    LB0[719] <= LB1[0];
end

always @(*) begin
    lb00 = LB0[0]; lb01 = LB1[0]; lb02 = LB2[0]; lb03 = LB3[0]; lb04 = LB4[0];
    lb10 = LB0[1]; lb11 = LB1[1]; lb12 = LB2[1]; lb13 = LB3[1]; lb14 = LB4[1];
    lb20 = LB0[2]; lb21 = LB1[2]; lb22 = LB2[2]; lb23 = LB3[2]; lb24 = LB4[2];
    lb30 = LB0[3]; lb31 = LB1[3]; lb32 = LB2[3]; lb33 = LB3[3]; lb34 = LB4[3];
    lb40 = LB0[4]; lb41 = LB1[4]; lb42 = LB2[4]; lb43 = LB3[4]; lb44 = LB4[4];
end

endmodule
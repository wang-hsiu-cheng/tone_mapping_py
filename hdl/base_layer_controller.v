module base_layer_controller (
    // Control signals
    input wire clk,

    // counter input
    input wire [10:0] w_counter,
    input wire [9:0]  h_counter,

    // Data input
    input wire signed [20:0] log_lum_out,

    // Data output
    output reg signed [20:0] base_layer_out
);

// Line buffer
wire signed [20:0] LB0 [0:4];
wire signed [20:0] LB1 [0:4];
wire signed [20:0] LB2 [0:4];
wire signed [20:0] LB3 [0:4];
wire signed [20:0] LB4 [0:4];

line_buffer line_buffer_u (
    .clk(clk),
    .log_lum_out(log_lum_out),
    .lb00(LB0[0]), .lb01(LB0[1]), .lb02(LB0[2]), .lb03(LB0[3]), .lb04(LB0[4]),
    .lb10(LB1[0]), .lb11(LB1[1]), .lb12(LB1[2]), .lb13(LB1[3]), .lb14(LB1[4]),
    .lb20(LB2[0]), .lb21(LB2[1]), .lb22(LB2[2]), .lb23(LB2[3]), .lb24(LB2[4]),
    .lb30(LB3[0]), .lb31(LB3[1]), .lb32(LB3[2]), .lb33(LB3[3]), .lb34(LB3[4]),
    .lb40(LB4[0]), .lb41(LB4[1]), .lb42(LB4[2]), .lb43(LB4[3]), .lb44(LB4[4])
);

// center position counter
// after (row, col) = (2,2) enter line buffer, center is at (0,0)
reg [10:0] w_center, w_center_next;
reg [9:0]  h_center, h_center_next;
always @(posedge clk) begin
    w_center <= w_center_next;
    h_center <= h_center_next;
end
always @(*) begin
    if (w_counter == 2 && h_counter == 1) begin
        h_center_next = 0;
        w_center_next = 0;
    end else begin
        if (h_center >= 719) begin
            h_center_next = 0;
            w_center_next = w_center + 1;
        end else begin
            h_center_next = h_center + 1;
            w_center_next = w_center;
        end
    end
end

// PE input selection (padding)
reg signed [20:0] pe_row0_in [0:4];
reg signed [20:0] pe_row1_in [0:4];
reg signed [20:0] pe_row2_in [0:4];
reg signed [20:0] pe_row3_in [0:4];
reg signed [20:0] pe_row4_in [0:4];
// pe_row0 (handle left & left-up & right & right-up padding)
always @(*) begin
    if (h_center == 0) begin
        if (w_center == 0) begin
            pe_row0_in[0] = LB2[2];
            pe_row0_in[1] = LB2[2];
            pe_row0_in[2] = LB2[2];
            pe_row0_in[3] = LB2[3];
            pe_row0_in[4] = LB2[4];
        end else if (w_center == 1) begin
            pe_row0_in[0] = LB2[1];
            pe_row0_in[1] = LB2[1];
            pe_row0_in[2] = LB2[2];
            pe_row0_in[3] = LB2[3];
            pe_row0_in[4] = LB2[4];
        end else if (w_center == 1278) begin
            pe_row0_in[0] = LB2[0];
            pe_row0_in[1] = LB2[1];
            pe_row0_in[2] = LB2[2];
            pe_row0_in[3] = LB2[3];
            pe_row0_in[4] = LB2[3];
        end else if (w_center == 1279) begin
            pe_row0_in[0] = LB2[0];
            pe_row0_in[1] = LB2[1];
            pe_row0_in[2] = LB2[2];
            pe_row0_in[3] = LB2[2];
            pe_row0_in[4] = LB2[2];
        end else begin
            pe_row0_in[0] = LB2[0];
            pe_row0_in[1] = LB2[1];
            pe_row0_in[2] = LB2[2];
            pe_row0_in[3] = LB2[3];
            pe_row0_in[4] = LB2[4];
        end
    end else if (h_center == 1) begin
        if (w_center == 0) begin
            pe_row0_in[0] = LB1[2];
            pe_row0_in[1] = LB1[2];
            pe_row0_in[2] = LB1[2];
            pe_row0_in[3] = LB1[3];
            pe_row0_in[4] = LB1[4];
        end else if (w_center == 1) begin
            pe_row0_in[0] = LB1[1];
            pe_row0_in[1] = LB1[1];
            pe_row0_in[2] = LB1[2];
            pe_row0_in[3] = LB1[3];
            pe_row0_in[4] = LB1[4];
        end else if (w_center == 1278) begin
            pe_row0_in[0] = LB1[0];
            pe_row0_in[1] = LB1[1];
            pe_row0_in[2] = LB1[2];
            pe_row0_in[3] = LB1[3];
            pe_row0_in[4] = LB1[3];
        end else if (w_center == 1279) begin
            pe_row0_in[0] = LB1[0];
            pe_row0_in[1] = LB1[1];
            pe_row0_in[2] = LB1[2];
            pe_row0_in[3] = LB1[2];
            pe_row0_in[4] = LB1[2];
        end else begin
            pe_row0_in[0] = LB1[0];
            pe_row0_in[1] = LB1[1];
            pe_row0_in[2] = LB1[2];
            pe_row0_in[3] = LB1[3];
            pe_row0_in[4] = LB1[4];
        end
    end else begin
        if (w_center == 0) begin
            pe_row0_in[0] = LB0[2];
            pe_row0_in[1] = LB0[2];
            pe_row0_in[2] = LB0[2];
            pe_row0_in[3] = LB0[3];
            pe_row0_in[4] = LB0[4];
        end else if (w_center == 1) begin
            pe_row0_in[0] = LB0[1];
            pe_row0_in[1] = LB0[1];
            pe_row0_in[2] = LB0[2];
            pe_row0_in[3] = LB0[3];
            pe_row0_in[4] = LB0[4];
        end else if (w_center == 1278) begin
            pe_row0_in[0] = LB0[0];
            pe_row0_in[1] = LB0[1];
            pe_row0_in[2] = LB0[2];
            pe_row0_in[3] = LB0[3];
            pe_row0_in[4] = LB0[3];
        end else if (w_center == 1279) begin
            pe_row0_in[0] = LB0[0];
            pe_row0_in[1] = LB0[1];
            pe_row0_in[2] = LB0[2];
            pe_row0_in[3] = LB0[2];
            pe_row0_in[4] = LB0[2];
        end else begin
            pe_row0_in[0] = LB0[0];
            pe_row0_in[1] = LB0[1];
            pe_row0_in[2] = LB0[2];
            pe_row0_in[3] = LB0[3];
            pe_row0_in[4] = LB0[4];
        end
    end
end
// pe_row1
always @(*) begin
    if (h_center == 0) begin
        if (w_center == 0) begin
            pe_row1_in[0] = LB2[2];
            pe_row1_in[1] = LB2[2];
            pe_row1_in[2] = LB2[2];
            pe_row1_in[3] = LB2[3];
            pe_row1_in[4] = LB2[4];
        end else if (w_center == 1) begin
            pe_row1_in[0] = LB2[1];
            pe_row1_in[1] = LB2[1];
            pe_row1_in[2] = LB2[2];
            pe_row1_in[3] = LB2[3];
            pe_row1_in[4] = LB2[4];
        end else if (w_center == 1278) begin
            pe_row1_in[0] = LB2[0];
            pe_row1_in[1] = LB2[1];
            pe_row1_in[2] = LB2[2];
            pe_row1_in[3] = LB2[3];
            pe_row1_in[4] = LB2[3];
        end else if (w_center == 1279) begin
            pe_row1_in[0] = LB2[0];
            pe_row1_in[1] = LB2[1];
            pe_row1_in[2] = LB2[2];
            pe_row1_in[3] = LB2[2];
            pe_row1_in[4] = LB2[2];
        end else begin
            pe_row1_in[0] = LB2[0];
            pe_row1_in[1] = LB2[1];
            pe_row1_in[2] = LB2[2];
            pe_row1_in[3] = LB2[3];
            pe_row1_in[4] = LB2[4];
        end
    end else begin
        if (w_center == 0) begin
            pe_row1_in[0] = LB1[2];
            pe_row1_in[1] = LB1[2];
            pe_row1_in[2] = LB1[2];
            pe_row1_in[3] = LB1[3];
            pe_row1_in[4] = LB1[4];
        end else if (w_center == 1) begin
            pe_row1_in[0] = LB1[1];
            pe_row1_in[1] = LB1[1];
            pe_row1_in[2] = LB1[2];
            pe_row1_in[3] = LB1[3];
            pe_row1_in[4] = LB1[4];
        end else if (w_center == 1278) begin
            pe_row1_in[0] = LB1[0];
            pe_row1_in[1] = LB1[1];
            pe_row1_in[2] = LB1[2];
            pe_row1_in[3] = LB1[3];
            pe_row1_in[4] = LB1[3];
        end else if (w_center == 1279) begin
            pe_row1_in[0] = LB1[0];
            pe_row1_in[1] = LB1[1];
            pe_row1_in[2] = LB1[2];
            pe_row1_in[3] = LB1[2];
            pe_row1_in[4] = LB1[2];
        end else begin
            pe_row1_in[0] = LB1[0];
            pe_row1_in[1] = LB1[1];
            pe_row1_in[2] = LB1[2];
            pe_row1_in[3] = LB1[3];
            pe_row1_in[4] = LB1[4];
        end
    end
end
// pe_row2
always @(*) begin
    if (w_center == 0) begin
        pe_row2_in[0] = LB2[2];
        pe_row2_in[1] = LB2[2];
        pe_row2_in[2] = LB2[2];
        pe_row2_in[3] = LB2[3];
        pe_row2_in[4] = LB2[4];
    end else if (w_center == 1) begin
        pe_row2_in[0] = LB2[1];
        pe_row2_in[1] = LB2[1];
        pe_row2_in[2] = LB2[2];
        pe_row2_in[3] = LB2[3];
        pe_row2_in[4] = LB2[4];
    end else if (w_center == 1278) begin
        pe_row2_in[0] = LB2[0];
        pe_row2_in[1] = LB2[1];
        pe_row2_in[2] = LB2[2];
        pe_row2_in[3] = LB2[3];
        pe_row2_in[4] = LB2[3];
    end else if (w_center == 1279) begin
        pe_row2_in[0] = LB2[0];
        pe_row2_in[1] = LB2[1];
        pe_row2_in[2] = LB2[2];
        pe_row2_in[3] = LB2[2];
        pe_row2_in[4] = LB2[2];
    end else begin
        pe_row2_in[0] = LB2[0];
        pe_row2_in[1] = LB2[1];
        pe_row2_in[2] = LB2[2];
        pe_row2_in[3] = LB2[3];
        pe_row2_in[4] = LB2[4];
    end
end
// pe_row3
always @(*) begin
    if (h_center == 719) begin
        if (w_center == 0) begin
            pe_row3_in[0] = LB2[2];
            pe_row3_in[1] = LB2[2];
            pe_row3_in[2] = LB2[2];
            pe_row3_in[3] = LB2[3];
            pe_row3_in[4] = LB2[4];
        end else if (w_center == 1) begin
            pe_row3_in[0] = LB2[1];
            pe_row3_in[1] = LB2[1];
            pe_row3_in[2] = LB2[2];
            pe_row3_in[3] = LB2[3];
            pe_row3_in[4] = LB2[4];
        end else if (w_center == 1278) begin
            pe_row3_in[0] = LB2[0];
            pe_row3_in[1] = LB2[1];
            pe_row3_in[2] = LB2[2];
            pe_row3_in[3] = LB2[3];
            pe_row3_in[4] = LB2[3];
        end else if (w_center == 1279) begin
            pe_row3_in[0] = LB2[0];
            pe_row3_in[1] = LB2[1];
            pe_row3_in[2] = LB2[2];
            pe_row3_in[3] = LB2[2];
            pe_row3_in[4] = LB2[2];
        end else begin
            pe_row3_in[0] = LB2[0];
            pe_row3_in[1] = LB2[1];
            pe_row3_in[2] = LB2[2];
            pe_row3_in[3] = LB2[3];
            pe_row3_in[4] = LB2[4];
        end
    end else begin
        if (w_center == 0) begin
            pe_row3_in[0] = LB3[2];
            pe_row3_in[1] = LB3[2];
            pe_row3_in[2] = LB3[2];
            pe_row3_in[3] = LB3[3];
            pe_row3_in[4] = LB3[4];
        end else if (w_center == 1) begin
            pe_row3_in[0] = LB3[1];
            pe_row3_in[1] = LB3[1];
            pe_row3_in[2] = LB3[2];
            pe_row3_in[3] = LB3[3];
            pe_row3_in[4] = LB3[4];
        end else if (w_center == 1278) begin
            pe_row3_in[0] = LB3[0];
            pe_row3_in[1] = LB3[1];
            pe_row3_in[2] = LB3[2];
            pe_row3_in[3] = LB3[3];
            pe_row3_in[4] = LB3[3];
        end else if (w_center == 1279) begin
            pe_row3_in[0] = LB3[0];
            pe_row3_in[1] = LB3[1];
            pe_row3_in[2] = LB3[2];
            pe_row3_in[3] = LB3[2];
            pe_row3_in[4] = LB3[2];
        end else begin
            pe_row3_in[0] = LB3[0];
            pe_row3_in[1] = LB3[1];
            pe_row3_in[2] = LB3[2];
            pe_row3_in[3] = LB3[3];
            pe_row3_in[4] = LB3[4];
        end
    end
end
// pe_row4 (handle left & left-botton & right & right-botton padding)
always @(*) begin
    if (h_center == 718) begin
        if (w_center == 0) begin
            pe_row4_in[0] = LB3[2];
            pe_row4_in[1] = LB3[2];
            pe_row4_in[2] = LB3[2];
            pe_row4_in[3] = LB3[3];
            pe_row4_in[4] = LB3[4];
        end else if (w_center == 1) begin
            pe_row4_in[0] = LB3[1];
            pe_row4_in[1] = LB3[1];
            pe_row4_in[2] = LB3[2];
            pe_row4_in[3] = LB3[3];
            pe_row4_in[4] = LB3[4];
        end else if (w_center == 1278) begin
            pe_row4_in[0] = LB3[0];
            pe_row4_in[1] = LB3[1];
            pe_row4_in[2] = LB3[2];
            pe_row4_in[3] = LB3[3];
            pe_row4_in[4] = LB3[3];
        end else if (w_center == 1279) begin
            pe_row4_in[0] = LB3[0];
            pe_row4_in[1] = LB3[1];
            pe_row4_in[2] = LB3[2];
            pe_row4_in[3] = LB3[2];
            pe_row4_in[4] = LB3[2];
        end else begin
            pe_row4_in[0] = LB3[0];
            pe_row4_in[1] = LB3[1];
            pe_row4_in[2] = LB3[2];
            pe_row4_in[3] = LB3[3];
            pe_row4_in[4] = LB3[4];
        end
    end else if (h_center == 719) begin
        if (w_center == 0) begin
            pe_row4_in[0] = LB2[2];
            pe_row4_in[1] = LB2[2];
            pe_row4_in[2] = LB2[2];
            pe_row4_in[3] = LB2[3];
            pe_row4_in[4] = LB2[4];
        end else if (w_center == 1) begin
            pe_row4_in[0] = LB2[1];
            pe_row4_in[1] = LB2[1];
            pe_row4_in[2] = LB2[2];
            pe_row4_in[3] = LB2[3];
            pe_row4_in[4] = LB2[4];
        end else if (w_center == 1278) begin
            pe_row4_in[0] = LB2[0];
            pe_row4_in[1] = LB2[1];
            pe_row4_in[2] = LB2[2];
            pe_row4_in[3] = LB2[3];
            pe_row4_in[4] = LB2[3];
        end else if (w_center == 1279) begin
            pe_row4_in[0] = LB2[0];
            pe_row4_in[1] = LB2[1];
            pe_row4_in[2] = LB2[2];
            pe_row4_in[3] = LB2[2];
            pe_row4_in[4] = LB2[2];
        end else begin
            pe_row4_in[0] = LB2[0];
            pe_row4_in[1] = LB2[1];
            pe_row4_in[2] = LB2[2];
            pe_row4_in[3] = LB2[3];
            pe_row4_in[4] = LB2[4];
        end
    end else begin
        if (w_center == 0) begin
            pe_row4_in[0] = LB4[2];
            pe_row4_in[1] = LB4[2];
            pe_row4_in[2] = LB4[2];
            pe_row4_in[3] = LB4[3];
            pe_row4_in[4] = LB4[4];
        end else if (w_center == 1) begin
            pe_row4_in[0] = LB4[1];
            pe_row4_in[1] = LB4[1];
            pe_row4_in[2] = LB4[2];
            pe_row4_in[3] = LB4[3];
            pe_row4_in[4] = LB4[4];
        end else if (w_center == 1278) begin
            pe_row4_in[0] = LB4[0];
            pe_row4_in[1] = LB4[1];
            pe_row4_in[2] = LB4[2];
            pe_row4_in[3] = LB4[3];
            pe_row4_in[4] = LB4[3];
        end else if (w_center == 1279) begin
            pe_row4_in[0] = LB4[0];
            pe_row4_in[1] = LB4[1];
            pe_row4_in[2] = LB4[2];
            pe_row4_in[3] = LB4[2];
            pe_row4_in[4] = LB4[2];
        end else begin
            pe_row4_in[0] = LB4[0];
            pe_row4_in[1] = LB4[1];
            pe_row4_in[2] = LB4[2];
            pe_row4_in[3] = LB4[3];
            pe_row4_in[4] = LB4[4];
        end
    end
end

// PE declaration
// row 0
wire signed [31:0] weighted_q_row0 [0:4];
wire [10:0] total_weight_row0 [0:4];
PE #(.DIST_EXP(0)) u_pe_00 (
    .clk(clk),
    .q_in(pe_row0_in[0]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row0[0]), 
    .total_weight_out(total_weight_row0[0])
);
PE #(.DIST_EXP(4)) u_pe_01 (
    .clk(clk),
    .q_in(pe_row0_in[1]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row0[1]), 
    .total_weight_out(total_weight_row0[1])
);
PE #(.DIST_EXP(29)) u_pe_02 (
    .clk(clk),
    .q_in(pe_row0_in[2]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row0[2]), 
    .total_weight_out(total_weight_row0[2])
);
PE #(.DIST_EXP(4)) u_pe_03 (
    .clk(clk),
    .q_in(pe_row0_in[3]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row0[3]), 
    .total_weight_out(total_weight_row0[3])
);
PE #(.DIST_EXP(0)) u_pe_04 (
    .clk(clk),
    .q_in(pe_row0_in[4]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row0[4]), 
    .total_weight_out(total_weight_row0[4])
);
// row 1
wire signed [31:0] weighted_q_row1 [0:4];
wire [10:0] total_weight_row1 [0:4];
PE #(.DIST_EXP(4)) u_pe_10 (
    .clk(clk),
    .q_in(pe_row1_in[0]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row1[0]), 
    .total_weight_out(total_weight_row1[0])
);
PE #(.DIST_EXP(421)) u_pe_11 (
    .clk(clk),
    .q_in(pe_row1_in[1]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row1[1]), 
    .total_weight_out(total_weight_row1[1])
);
PE #(.DIST_EXP(820)) u_pe_12 (
    .clk(clk),
    .q_in(pe_row1_in[2]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row1[2]), 
    .total_weight_out(total_weight_row1[2])
);
PE #(.DIST_EXP(421)) u_pe_13 (
    .clk(clk),
    .q_in(pe_row1_in[3]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row1[3]), 
    .total_weight_out(total_weight_row1[3])
);
PE #(.DIST_EXP(4)) u_pe_14 (
    .clk(clk),
    .q_in(pe_row1_in[4]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row1[4]), 
    .total_weight_out(total_weight_row1[4])
);
// row 2
wire signed [31:0] weighted_q_row2 [0:4];
wire [10:0] total_weight_row2 [0:4];
PE #(.DIST_EXP(29)) u_pe_20 (
    .clk(clk),
    .q_in(pe_row2_in[0]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row2[0]), 
    .total_weight_out(total_weight_row2[0])
);
PE #(.DIST_EXP(820)) u_pe_21 (
    .clk(clk),
    .q_in(pe_row2_in[1]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row2[1]), 
    .total_weight_out(total_weight_row2[1])
);
PE #(.DIST_EXP(1024)) u_pe_22 (
    .clk(clk),
    .q_in(pe_row2_in[2]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row2[2]), 
    .total_weight_out(total_weight_row2[2])
);
PE #(.DIST_EXP(820)) u_pe_23 (
    .clk(clk),
    .q_in(pe_row2_in[3]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row2[3]), 
    .total_weight_out(total_weight_row2[3])
);
PE #(.DIST_EXP(29)) u_pe_24 (
    .clk(clk),
    .q_in(pe_row2_in[4]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row2[4]), 
    .total_weight_out(total_weight_row2[4])
);
// row 3
wire signed [31:0] weighted_q_row3 [0:4];
wire [10:0] total_weight_row3 [0:4];
PE #(.DIST_EXP(4)) u_pe_30 (
    .clk(clk),
    .q_in(pe_row3_in[0]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row3[0]), 
    .total_weight_out(total_weight_row3[0])
);
PE #(.DIST_EXP(421)) u_pe_31 (
    .clk(clk),
    .q_in(pe_row3_in[1]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row3[1]), 
    .total_weight_out(total_weight_row3[1])
);
PE #(.DIST_EXP(820)) u_pe_32 (
    .clk(clk),
    .q_in(pe_row3_in[2]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row3[2]), 
    .total_weight_out(total_weight_row3[2])
);
PE #(.DIST_EXP(421)) u_pe_33 (
    .clk(clk),
    .q_in(pe_row3_in[3]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row3[3]), 
    .total_weight_out(total_weight_row3[3])
);
PE #(.DIST_EXP(4)) u_pe_34 (
    .clk(clk),
    .q_in(pe_row3_in[4]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row3[4]), 
    .total_weight_out(total_weight_row3[4])
);
// row 4
wire signed [31:0] weighted_q_row4 [0:4];
wire [10:0] total_weight_row4 [0:4];
PE #(.DIST_EXP(0)) u_pe_40 (
    .clk(clk),
    .q_in(pe_row4_in[0]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row4[0]), 
    .total_weight_out(total_weight_row4[0])
);
PE #(.DIST_EXP(4)) u_pe_41 (
    .clk(clk),
    .q_in(pe_row4_in[1]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row4[1]), 
    .total_weight_out(total_weight_row4[1])
);
PE #(.DIST_EXP(29)) u_pe_42 (
    .clk(clk),
    .q_in(pe_row4_in[2]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row4[2]), 
    .total_weight_out(total_weight_row4[2])
);
PE #(.DIST_EXP(4)) u_pe_43 (
    .clk(clk),
    .q_in(pe_row4_in[3]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row4[3]), 
    .total_weight_out(total_weight_row4[3])
);
PE #(.DIST_EXP(0)) u_pe_44 (
    .clk(clk),
    .q_in(pe_row4_in[4]),
    .p_in(pe_row2_in[2]),
    .weighted_q(weighted_q_row4[4]), 
    .total_weight_out(total_weight_row4[4])
);

reg signed [34:0] numerator_0 [0:4]; // Q11.24
reg signed [34:0] numerator_1;       // Q11.24
reg [12:0] denominator_0 [0:4]; // UQ3.10
reg [12:0] denominator_1;

// stage 1 adder
always @(posedge clk) begin
    numerator_0[0] <= weighted_q_row0[0] + weighted_q_row0[1] + weighted_q_row0[2] + weighted_q_row0[3] + weighted_q_row0[4];
    numerator_0[1] <= weighted_q_row1[0] + weighted_q_row1[1] + weighted_q_row1[2] + weighted_q_row1[3] + weighted_q_row1[4];
    numerator_0[2] <= weighted_q_row2[0] + weighted_q_row2[1] + weighted_q_row2[2] + weighted_q_row2[3] + weighted_q_row2[4];
    numerator_0[3] <= weighted_q_row3[0] + weighted_q_row3[1] + weighted_q_row3[2] + weighted_q_row3[3] + weighted_q_row3[4];
    numerator_0[4] <= weighted_q_row4[0] + weighted_q_row4[1] + weighted_q_row4[2] + weighted_q_row4[3] + weighted_q_row4[4];

    denominator_0[0] <= total_weight_row0[0] + total_weight_row0[1] + total_weight_row0[2] + total_weight_row0[3] + total_weight_row0[4];
    denominator_0[1] <= total_weight_row1[0] + total_weight_row1[1] + total_weight_row1[2] + total_weight_row1[3] + total_weight_row1[4];
    denominator_0[2] <= total_weight_row2[0] + total_weight_row2[1] + total_weight_row2[2] + total_weight_row2[3] + total_weight_row2[4];
    denominator_0[3] <= total_weight_row3[0] + total_weight_row3[1] + total_weight_row3[2] + total_weight_row3[3] + total_weight_row3[4];
    denominator_0[4] <= total_weight_row4[0] + total_weight_row4[1] + total_weight_row4[2] + total_weight_row4[3] + total_weight_row4[4];
end

// stage 2 adder
always @(posedge clk) begin
    numerator_1 <= numerator_0[0] + numerator_0[1] + numerator_0[2] + numerator_0[3] + numerator_0[4];

    denominator_1 <= denominator_0[0] + denominator_0[1] + denominator_0[2] + denominator_0[3] + denominator_0[4];
end

// stage 3 lut
wire [9:0] wp_lut_data; // Q0.10
reg [9:0] wp_lut_reg;
wp_divide_lut u_wp_divide_lut (
    .addr(denominator_1),
    .data(wp_lut_data)
);
reg signed [34:0] numerator_1_reg;    // Q11.24
always @(posedge clk) begin
    wp_lut_reg <= wp_lut_data;
    numerator_1_reg <= numerator_1;
end

// stage 4 multiply stage 1
wire signed [10:0]wp_lut_signed;
assign wp_lut_signed = {1'b0, wp_lut_reg};
reg signed [44:0] b_val_stage1; // Q11.34
always @(posedge clk) begin
    b_val_stage1 <= wp_lut_signed * numerator_1_reg;
end

// stage 5 mutiply stage 2
reg signed [44:0] b_val_stage2; // Q11.34
always @(posedge clk) begin
    b_val_stage2 <= b_val_stage1;
end

// stage 6 output
reg signed [24:0] b_quant; // Q11.14
reg signed [20:0] b_clamp; // base_layer_out Q7.14
always @(*) begin
    b_quant = b_val_stage2[44:20];
    if (b_quant > 1048575)
        b_clamp = 1048575;
    else if (b_quant < -1048576)
        b_clamp = -1048576;
    else
        b_clamp = b_quant[20:0];
end
always @(posedge clk) begin
    base_layer_out <= b_clamp;
end

endmodule
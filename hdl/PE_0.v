module PE # (
    parameter DIST_EXP = 1024 // Distance sqaure gaussion value in UQ1.10
) (
    // Control signals
    input wire clk,

    // Data input
    input wire signed [20:0] q_in,
    input wire signed [20:0] p_in,

    // Data output
    output reg signed [31:0] weighted_q, // Q8.24
    output reg [10:0] total_weight_out   // UQ1.10
);

// stage 1
reg signed [21:0] diff_comb;        //  Q8.14
reg [20:0] diff_abs, diff_abs_next; // UQ7.14
always @(posedge clk) begin
    diff_abs <= diff_abs_next;
end
always @(*) begin
    diff_comb = p_in - q_in;
    if (diff_comb[21]) begin
        diff_abs_next = -diff_comb;
    end else begin
        diff_abs_next = diff_comb[20:0];
    end
end
reg signed [20:0] q_reg_stage1; // pipeline register
always @(posedge clk) begin
    q_reg_stage1 <= q_in;
end

// stage 2
reg [41:0] diff_sq_comb;    // UQ14.28
reg [24:0] diff_sq_quant;   // UQ14.11
reg [13:0] range_exp_index, range_exp_index_reg; // UQ3.11
always @(*) begin
    diff_sq_comb = diff_abs * diff_abs;
    // quant
    diff_sq_quant = diff_sq_comb[41:17];
    // clamp
    if (diff_sq_quant > 16383)
        range_exp_index = 16383;
    else
        range_exp_index = diff_sq_quant[13:0];
end
always @(posedge clk) begin
    range_exp_index_reg <= range_exp_index;
end
reg signed [20:0] q_reg_stage2; // pipeline register
always @(posedge clk) begin
    q_reg_stage2 <= q_reg_stage1;
end


// stage 3
wire [10:0] range_exp_weight; // UQ1.10
reg  [10:0] space_exp_weight; // UQ1.10
always @(posedge clk) begin
    space_exp_weight <= DIST_EXP;
end
range_exp_lut range_exp_lut_u (
    .addr(range_exp_index_reg),
    .data(range_exp_weight)
);
reg [20:0] total_weight; // UQ1.20
reg [10:0] total_weight_quant; // UQ1.10
always @(*) begin
    total_weight = range_exp_weight * space_exp_weight;
end
always @(posedge clk) begin
    total_weight_quant <= total_weight[20:10];
end
reg signed [20:0] q_reg_stage3; // pipeline register
always @(posedge clk) begin
    q_reg_stage3 <= q_reg_stage2;
end

// stage 4
wire signed [11:0] signed_weight;
assign signed_weight = {1'b0, total_weight_quant};
always @(posedge clk) begin
    weighted_q <= q_reg_stage3 * signed_weight;
    total_weight_out <= total_weight_quant;
end

endmodule
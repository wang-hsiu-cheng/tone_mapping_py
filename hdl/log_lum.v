module log_lum (
    input wire clk,

    // input wire valid_in,
    input wire [7:0] R,
    input wire [7:0] G,
    input wire [7:0] B,
    input wire [7:0] E,

    // output wire valid_out,
    output reg signed [20:0] log_lum_out
);

// stage 1
reg [15:0] Lm, Lm_next;
always @(posedge clk) begin
    Lm <= Lm_next;
end
reg [13:0] Rw;
reg [15:0] Gw;
reg [12:0] Bw;
always @(*) begin
    Rw = R * 6'd54;
    Gw = G * 8'd183;
    Bw = B * 5'd19;
    Lm_next = Rw + Gw + Bw;
end
reg [7:0] E_reg;
always @(posedge clk) begin
    E_reg <= E;
end

// stage 2
reg [3:0] Msb_next;

always @(*) begin
    if (Lm[15]) begin
        Msb_next = 4'd15;
    end else if (Lm[14]) begin
        Msb_next = 4'd14;
    end else if (Lm[13]) begin
        Msb_next = 4'd13;
    end else if (Lm[12]) begin
        Msb_next = 4'd12;
    end else if (Lm[11]) begin
        Msb_next = 4'd11;
    end else if (Lm[10]) begin
        Msb_next = 4'd10;
    end else if (Lm[9]) begin
        Msb_next = 4'd9;
    end else if (Lm[8]) begin
        Msb_next = 4'd8;
    end else if (Lm[7]) begin
        Msb_next = 4'd7;
    end else if (Lm[6]) begin
        Msb_next = 4'd6;
    end else if (Lm[5]) begin
        Msb_next = 4'd5;
    end else if (Lm[4]) begin
        Msb_next = 4'd4;
    end else if (Lm[3]) begin
        Msb_next = 4'd3;
    end else if (Lm[2]) begin
        Msb_next = 4'd2;
    end else if (Lm[1]) begin
        Msb_next = 4'd1;
    end else begin
        Msb_next = 4'd0;
    end
end

wire [25:0] extend_lm; // 15 bit Lm + 11 bit pad 0
assign extend_lm = {Lm[14:0], {11{1'b0}}};

reg [11:0] idx;
always @(*) begin
    if (Msb_next == 4'd15) begin
        idx = extend_lm[25-:12];
    end else if (Msb_next == 4'd14) begin
        idx = extend_lm[24-:12];
    end else if (Msb_next == 4'd13) begin
        idx = extend_lm[23-:12];
    end else if (Msb_next == 4'd12) begin
        idx = extend_lm[22-:12];
    end else if (Msb_next == 4'd11) begin
        idx = extend_lm[21-:12];
    end else if (Msb_next == 4'd10) begin
        idx = extend_lm[20-:12];
    end else if (Msb_next == 4'd9) begin
        idx = extend_lm[19-:12];
    end else if (Msb_next == 4'd8) begin
        idx = extend_lm[18-:12];
    end else if (Msb_next == 4'd7) begin
        idx = extend_lm[17-:12];
    end else if (Msb_next == 4'd6) begin
        idx = extend_lm[16-:12];
    end else if (Msb_next == 4'd5) begin
        idx = extend_lm[15-:12];
    end else if (Msb_next == 4'd4) begin
        idx = extend_lm[14-:12];
    end else if (Msb_next == 4'd3) begin
        idx = extend_lm[13-:12];
    end else if (Msb_next == 4'd2) begin
        idx = extend_lm[12-:12];
    end else if (Msb_next == 4'd1) begin
        idx = extend_lm[11-:12];
    end else begin
        idx = 12'd0;
    end
end

wire [12:0] base;
reg  [12:0] base_reg;
lm_base_lut u_lm_base_lut (
    .idx(idx),
    .base(base)
);
always @(posedge clk) begin
    base_reg <= base;
end


reg signed [8:0] exp_value, exp_value_next;
always @(posedge clk) begin
    exp_value <= exp_value_next;
end
always @(*) begin
    exp_value_next = E_reg - 9'sd144 + Msb_next;
end

// stage 3
reg signed [21:0] exp_log;
reg signed [21:0] log_lum;
always @(*) begin
    exp_log = exp_value * 14'sd4932;
    log_lum = base_reg + exp_log;
end

always @(*) begin
    log_lum_out = {log_lum[21], log_lum[19:0]};
end

endmodule
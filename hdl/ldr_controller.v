module ldr_controller #(
    parameter CH_NUM = 4,    
    parameter BW_PER_CH = 8,
    parameter HEIGHT = 720,
    parameter WIDTH = 1280,
    parameter DATA_WIDTH_L = 21
) (
    input wire clk,
    input wire srst_n,

    // Controll signal
    input wire enable,
    output reg done,

    // B range
    input wire signed [20:0] b_max,
    input wire signed [20:0] b_min,

    // input sram interface
    input wire [CH_NUM*BW_PER_CH-1:0] sram_rdata_in,
    output reg [20-1:0] sram_addr_in,

    // act sram L interface
    input wire [DATA_WIDTH_L-1:0] sram_rdata_l,
    output reg [20-1:0] sram_addr_l,

    // act sram B interface
    input wire [DATA_WIDTH_L-1:0] sram_rdata_b,
    output reg [20-1:0] sram_addr_b,

    // SRAM_OUT Interface (Write Final RGB)
    output reg [23:0] sram_wdata_out,
    output reg sram_wen_out,
    output reg [20-1:0] sram_addr_out
);

// fsm 
localparam IDLE = 3'd0;
localparam READ = 3'd1;
localparam OUT  = 3'd2;
localparam DONE = 3'd7;

reg [2:0] state, next_state;
always @(posedge clk) begin
    if (~srst_n) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

// get input sram data
reg [10:0] w_counter, w_counter_next; // log2(1280) = 10.3
reg  [9:0] h_counter, h_counter_next; // log2(720)  = 9.4
always @(posedge clk) begin
    h_counter <= h_counter_next;
    w_counter <= w_counter_next;
end
always @(*) begin
    case(state)
        IDLE: begin
            h_counter_next = 0;
            w_counter_next = 0;
        end
        READ: begin
            if (w_counter >= 1279) begin
                h_counter_next = h_counter + 1;
                w_counter_next = 0;
            end else begin
                h_counter_next = h_counter;
                w_counter_next = w_counter + 1;
            end
        end
        OUT: begin
            if (w_counter >= 1279) begin
                h_counter_next = h_counter + 1;
                w_counter_next = 0;
            end else begin
                h_counter_next = h_counter;
                w_counter_next = w_counter + 1;
            end
        end
        default: begin
            h_counter_next = h_counter;
            w_counter_next = w_counter;
        end
    endcase
end
always @(posedge clk) begin
    if (~srst_n) begin
        sram_addr_l <= 0;
        sram_addr_b <= 0;
        sram_addr_in <= 0;
    end else begin
        sram_addr_l <= h_counter_next * 1280 + w_counter_next;
        sram_addr_b <= h_counter_next * 1280 + w_counter_next;
        sram_addr_in <= h_counter_next * 1280 + w_counter_next;
    end
end

reg signed [20-1:0] base_reg;
reg signed [20-1:0] lm_reg;
reg [BW_PER_CH-1:0] R_reg;
reg [BW_PER_CH-1:0] G_reg;
reg [BW_PER_CH-1:0] B_reg;
reg [BW_PER_CH-1:0] E_reg;

// stage 1
always @(posedge clk) begin
    if (~srst_n) begin
        base_reg <= 0;
        lm_reg <= 0;
        R_reg <= 0;
        G_reg <= 0;
        B_reg <= 0;
        E_reg <= 0;
    end else begin
        base_reg <= sram_rdata_b[20:0];
        lm_reg   <= sram_rdata_l[20:0];
        R_reg <= sram_rdata_in[31:24];
        G_reg <= sram_rdata_in[23:16];
        B_reg <= sram_rdata_in[15:8];
        E_reg <= sram_rdata_in[7:0];
    end
end

reg signed [20:0] b_range; // Q7.14 - Q7.14 = UQ7.14 enough
always @(posedge clk) begin
    b_range <= b_max - b_min;
end

// stage 2
wire [11:0] k_lut_addr; // Q6.6
assign k_lut_addr = b_range[19:8]; 
wire [18-1:0] k_lut_data; // Q6.12 * 2 => Q7.11
k_divide_lut u_k_divide_lut (
    .addr(k_lut_addr), 
    .data(k_lut_data)
);
wire signed [18:0] K_lut_signed = {1'b0, k_lut_data};

reg signed [39-1:0] b_compress_reg; // Q7.14 * UQ7.11 = Q14.25
always @(posedge clk) begin
    b_compress_reg <= K_lut_signed * base_reg;
end
reg signed [20-1:0] lm_stage1;
always @(posedge clk) begin
    lm_stage1 <= lm_reg;
end
reg [BW_PER_CH-1:0] R_stage1;
reg [BW_PER_CH-1:0] G_stage1;
reg [BW_PER_CH-1:0] B_stage1;
reg [BW_PER_CH-1:0] E_stage1;
always @(posedge clk) begin
    R_stage1 <= R_reg;
    G_stage1 <= G_reg;
    B_stage1 <= B_reg;
    E_stage1 <= E_reg;
end
reg [20:0] detail;
always @(posedge clk) begin
    detail <= lm_reg - base_reg;
end

// stage 3
reg signed [28-1:0] b_compress_quant; // Q14.14
reg signed [21-1:0] b_compress_clamp; // Q7.14
always @(*) begin
    b_compress_quant = b_compress_reg[38:11];
    if (b_compress_quant > 1048575)
        b_compress_clamp = 1048575;
    else if (b_compress_quant < -1048576)
        b_compress_clamp = -1048576;
    else
        b_compress_clamp = b_compress_quant[20:0];
end
reg signed [20:0] I_prime;
always @(posedge clk) begin
    I_prime <= b_compress_clamp + $signed({1'b0, detail}) - lm_stage1;
end
reg [BW_PER_CH-1:0] R_stage2;
reg [BW_PER_CH-1:0] G_stage2;
reg [BW_PER_CH-1:0] B_stage2;
reg [BW_PER_CH-1:0] E_stage2;
always @(posedge clk) begin
    R_stage2 <= R_stage1;
    G_stage2 <= G_stage1;
    B_stage2 <= B_stage1;
    E_stage2 <= E_stage1;
end

// stage 4
reg signed [37:0] I_ratio_comb; // Q7.14 * UQ2.15 = Q9.29
reg signed [22:0] I_ratio_quant; // Q9.14;

always @(*) begin
    I_ratio_comb = I_prime * 18'sd108853;
end
always @(posedge clk) begin
    I_ratio_quant <= I_ratio_comb[37:15];
end
reg [BW_PER_CH-1:0] R_stage3;
reg [BW_PER_CH-1:0] G_stage3;
reg [BW_PER_CH-1:0] B_stage3;
reg [BW_PER_CH-1:0] E_stage3;
always @(posedge clk) begin
    R_stage3 <= R_stage2;
    G_stage3 <= G_stage2;
    B_stage3 <= B_stage2;
    E_stage3 <= E_stage2;
end

// stage 5
reg signed [8:0] I_int;   // Q9.0
reg [13:0] I_frac; // Q0.14
always @(*) begin
    I_int  = I_ratio_quant[22:14];
    I_frac = I_ratio_quant[13:0];
end

wire [11:0] power_lut_addr = I_frac[13:2];
wire [12:0] power_lut_data; // Q1.12

power_lut u_power_lut (
    .addr(power_lut_addr),
    .data(power_lut_data)
);

reg signed [5:0] total_shift;
reg [20:0] R_tmp; // UQ8.0 * UQ1.12 -> UQ9.12
reg [20:0] G_tmp; // UQ8.0 * UQ1.12 -> UQ9.12
reg [20:0] B_tmp; // UQ8.0 * UQ1.12 -> UQ9.12
always @(posedge clk) begin
    total_shift <= $signed({1'b0, E_stage3}) - $signed(8'd140) + I_int;
    R_tmp <= R_stage3 * power_lut_data;
    G_tmp <= G_stage3 * power_lut_data;
    B_tmp <= B_stage3 * power_lut_data;
end

// --- Stage 6: Shifter and Saturation ---

reg [7:0] R_final, G_final, B_final;

// 為了防止左移時溢位（原本 21 bit + 最大左移 15 bit = 36 bit）
// 我們定義一個足夠寬的暫存器來存到位移後的結果
reg [35:0] r_shifted, g_shifted, b_shifted;

// 取得位移絕對值
wire [4:0] abs_shift = (total_shift[5]) ? (-total_shift[4:0]) : total_shift[4:0];
wire shift_dir = total_shift[5]; // 1 為右移 (負), 0 為左移 (正)

always @(*) begin
    // R 通道位移邏輯
    if (shift_dir == 1'b0) // 正：左移 (Left Shift)
        r_shifted = {15'b0, R_tmp} << abs_shift;
    else                   // 負：右移 (Right Shift)
        r_shifted = {15'b0, R_tmp} >> abs_shift;

    // G 通道位移邏輯
    if (shift_dir == 1'b0)
        g_shifted = {15'b0, G_tmp} << abs_shift;
    else
        g_shifted = {15'b0, G_tmp} >> abs_shift;

    // B 通道位移邏輯
    if (shift_dir == 1'b0)
        b_shifted = {15'b0, B_tmp} << abs_shift;
    else
        b_shifted = {15'b0, B_tmp} >> abs_shift;
end

// --- 最終飽和處理 (Clipping) 與 輸出暫存器 ---
always @(posedge clk) begin
    // 檢查 8-bit 以上的位元是否為 1
    // 如果 [35:8] 任何一位是 1，代表數值 > 255
    if (|r_shifted[35:8]) 
        R_final <= 8'd255;
    else
        R_final <= r_shifted[7:0];

    if (|g_shifted[35:8]) 
        G_final <= 8'd255;
    else
        G_final <= g_shifted[7:0];

    if (|b_shifted[35:8]) 
        B_final <= 8'd255;
    else
        B_final <= b_shifted[7:0];
end

// stage 7 output sram
reg [10:0] w_out, w_out_next; // log2(1280) = 10.3
reg  [9:0] h_out, h_out_next; // log2(720)  = 9.4
always @(posedge clk) begin
    h_out <= h_out_next;
    w_out <= w_out_next;
end
always @(*) begin
    case(state)
        IDLE: begin
            h_out_next = 0;
            w_out_next = 0;
        end
        READ: begin
            h_out_next = 0;
            w_out_next = 0;
        end
        OUT: begin
            if (w_out >= 1279) begin
                h_out_next = h_out + 1;
                w_out_next = 0;
            end else begin
                h_out_next = h_out;
                w_out_next = w_out + 1;
            end
        end
        default: begin
            h_out_next = h_out;
            w_out_next = w_out;
        end
    endcase
end
// sram output
always @(posedge clk) begin
    if (~srst_n) begin
        sram_addr_out <= 0;
    end else begin
        sram_addr_out <= h_out_next * 1280 + w_out_next;
        if (next_state == OUT) begin
            sram_wen_out <= 0;
            sram_wdata_out <= {R_final, G_final, B_final};
        end else begin
            sram_wen_out <= 1;
            sram_wdata_out <= 0;
        end
    end
end

// state transition logic
always @(*) begin
    case(state)
        IDLE: begin
            if (enable) next_state = READ;
            else next_state = state;
        end
        READ: begin
            if (w_counter >= 7) next_state = OUT;
            else next_state = state;
        end
        OUT: begin
            if (w_out >= 1279 && h_out >= 719) next_state = DONE;
            else next_state = state;
        end
        DONE: begin
            next_state = IDLE;
        end
        default: begin
            next_state = IDLE;
        end
    endcase
end

always @(posedge clk) begin
    if (next_state == DONE) begin
        done <= 1;
    end else begin
        done <= 0;
    end
end

endmodule
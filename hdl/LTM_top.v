module LTM_top #(
    parameter CH_NUM = 4,    
    parameter BW_PER_CH = 8,
    parameter HEIGHT = 720,
    parameter WIDTH = 1280,
    parameter DATA_WIDTH_L = 21
) (
    // Control signals
    input wire clk,
    input wire srst_n,
    input wire enable,
    output reg valid,

    // SRAM read data inputs
    input wire [CH_NUM*BW_PER_CH-1:0] sram_rdata_in,
    input wire [DATA_WIDTH_L-1:0] sram_rdata_l,
    input wire [DATA_WIDTH_L-1:0] sram_rdata_b,

    // SRAM address outputs
    output reg [20-1:0] sram_addr_in,
    output reg [20-1:0] sram_addr_l,
    output reg [20-1:0] sram_addr_b,
    output wire [20-1:0] sram_addr_out,

    // SRAM write enale
    output reg sram_wen_l,
    output reg sram_wen_b,
    output wire sram_wen_out,

    // SRAM output data
    output wire [DATA_WIDTH_L-1:0] sram_wdata_l,
    output wire [DATA_WIDTH_L-1:0] sram_wdata_b,
    output wire [(CH_NUM-1)*BW_PER_CH-1:0] sram_wdata_out
);

// FSM
localparam IDLE = 4'd0;
localparam OP   = 4'd1;
localparam LDR  = 4'd2;
localparam DONE = 4'd15;

reg [3:0] state, next_state;
wire [19:0] ll_addr_in, ll_addr_l, ll_addr_b;
wire ll_wen_l, ll_wen_b;
wire [19:0] ldr_addr_in;
wire [19:0] ldr_addr_l; // Reads from L (Log Lum)
wire [19:0] ldr_addr_b; // Reads from B
wire [19:0] total_pixels = HEIGHT * WIDTH;

always @(posedge clk) begin
    if (~srst_n) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

reg enable_reg;
always @(posedge clk) begin
    enable_reg <= enable;
end

// log_lum controller
wire log_lum_done;
log_lum_controller log_lum_controller_u (
    .clk(clk),
    .srst_n(srst_n),
    .enable(state == IDLE && next_state == OP),
    .done(log_lum_done),
    // input sram interface
    .sram_rdata_in(sram_rdata_in),
    .sram_addr_in(ll_addr_in),
    // act sram L interface
    .sram_rdata_l(sram_rdata_l),
    .sram_addr_l(ll_addr_l),
    .sram_wen_l(ll_wen_l),
    .sram_wdata_l(sram_wdata_l),
    // act sram B interface
    .sram_rdata_b(sram_rdata_b),
    .sram_addr_b(ll_addr_b),
    .sram_wen_b(ll_wen_b),
    .sram_wdata_b(sram_wdata_b)
);

wire ldr_done;
ldr_controller ldr_controller_u (
    .clk(clk),
    .srst_n(srst_n),
    .enable(state == OP && next_state == LDR),
    .total_pixels(total_pixels),
    .done(ldr_done),
    .sram_addr_in(ldr_addr_in),
    .sram_rdata_in(sram_rdata_in), 
    // act sram L interface
    .sram_rdata_l(sram_rdata_l),
    .sram_addr_l(ldr_addr_l),
    // act sram B interface
    .sram_rdata_b(sram_rdata_b),
    .sram_addr_b(ldr_addr_b),
    .sram_addr_out(sram_addr_out),
    .sram_wdata_out(sram_wdata_out),
    .sram_wen_out(sram_wen_out)
);

always @(*) begin
    case(state)
        IDLE: begin
            if (enable_reg) next_state = OP;
            else next_state = state;
        end
        OP: begin
            if (log_lum_done) next_state = LDR;
            else next_state = state;
        end
        LDR: begin
            if (ldr_done) next_state = DONE;
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
        valid <= 1;
    end else begin
        valid <= 0;
    end
end

// MUX Logic (Shared Resources Control)

// SRAM_IN, SRAM_L, SRAM_B 的位址與寫入控制需根據狀態切換
always @(*) begin
    if (state == OP) begin
        // [OP State] 由 Log Lum Controller 控制
        sram_addr_in = ll_addr_in;
        sram_addr_l  = ll_addr_l;
        sram_addr_b  = ll_addr_b;
        sram_wen_l   = ll_wen_l;
        sram_wen_b   = ll_wen_b;
    end else if (state == LDR) begin
        // [LDR State] 由 Tone Mapping Controller 控制
        sram_addr_in = ldr_addr_in; // 讀取原始影像
        sram_addr_l  = ldr_addr_l;  // 讀取 L
        sram_addr_b  = ldr_addr_b;  // 讀取 B
        sram_wen_l   = 1'b1;       // 此階段只讀不寫
        sram_wen_b   = 1'b1;       // 此階段只讀不寫
    end else begin
        // IDLE / DONE
        sram_addr_in = 20'd0;
        sram_addr_l  = 20'd0;
        sram_addr_b  = 20'd0;
        sram_wen_l   = 1'b1;
        sram_wen_b   = 1'b1;
    end
end

endmodule
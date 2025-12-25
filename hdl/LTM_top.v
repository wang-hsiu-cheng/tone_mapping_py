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
    output wire sram_wen_l,
    output wire sram_wen_b,
    output wire sram_wen_out,

    // SRAM output data
    output wire [DATA_WIDTH_L-1:0] sram_wdata_l,
    output wire [DATA_WIDTH_L-1:0] sram_wdata_b,
    output wire [24-1:0] sram_wdata_out
);

// FSM
localparam IDLE = 4'd0;
localparam OP   = 4'd1;
localparam LDR  = 4'd2;
localparam DONE = 4'd15;

reg [3:0] state, next_state;
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

// b range
wire signed [20:0] b_max_signal;
wire signed [20:0] b_min_signal;

// sram mux
// sram address
wire [20-1:0] sram_addr_in_op;
wire [20-1:0] sram_addr_l_op;
wire [20-1:0] sram_addr_b_op;

wire [20-1:0] sram_addr_in_ldr;
wire [20-1:0] sram_addr_l_ldr;
wire [20-1:0] sram_addr_b_ldr;


always @(*) begin
    if (next_state == LDR) begin
        sram_addr_in = sram_addr_in_ldr;
        sram_addr_l = sram_addr_l_ldr;
        sram_addr_b = sram_addr_b_ldr;
    end else begin
        sram_addr_in = sram_addr_in_op;
        sram_addr_l = sram_addr_l_op;
        sram_addr_b = sram_addr_b_op;
    end
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
    .sram_addr_in(sram_addr_in_op),
    // act sram L interface
    .sram_addr_l(sram_addr_l_op),
    .sram_wen_l(sram_wen_l),
    .sram_wdata_l(sram_wdata_l),
    // act sram B interface
    .sram_addr_b(sram_addr_b_op),
    .sram_wen_b(sram_wen_b),
    .sram_wdata_b(sram_wdata_b),
    // b range
    .b_max(b_max_signal),
    .b_min(b_min_signal)
);

// ldr controller
wire ldr_done;

ldr_controller ldr_controller_u (
    .clk(clk),
    .srst_n(srst_n),
    // Controll signal
    .enable(state == OP && next_state == LDR),
    .done(ldr_done),
    // B range
    .b_max(b_max_signal),
    .b_min(b_min_signal),
    // input sram interface
    .sram_addr_in(sram_addr_in_ldr),
    .sram_rdata_in(sram_rdata_in), 
    // act sram L interface
    .sram_rdata_l(sram_rdata_l),
    .sram_addr_l(sram_addr_l_ldr),
    // act sram B interface
    .sram_rdata_b(sram_rdata_b),
    .sram_addr_b(sram_addr_b_ldr),
    //output sram
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

endmodule
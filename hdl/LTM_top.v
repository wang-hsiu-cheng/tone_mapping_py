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
    output wire [20-1:0] sram_addr_in,
    output wire [20-1:0] sram_addr_l,
    output wire [20-1:0] sram_addr_b,

    // SRAM write enale
    output wire sram_wen_l,
    output wire sram_wen_b,

    // SRAM output data
    output wire [DATA_WIDTH_L-1:0] sram_wdata_l,
    output wire [DATA_WIDTH_L-1:0] sram_wdata_b
);

// FSM
localparam IDLE = 4'd0;
localparam OP   = 4'd1;
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

// log_lum controller
wire log_lum_done;
log_lum_controller log_lum_controller_u (
    .clk(clk),
    .srst_n(srst_n),
    .enable(state == IDLE && next_state == OP),
    .done(log_lum_done),
    // input sram interface
    .sram_rdata_in(sram_rdata_in),
    .sram_addr_in(sram_addr_in),
    // act sram L interface
    .sram_rdata_l(sram_rdata_l),
    .sram_addr_l(sram_addr_l),
    .sram_wen_l(sram_wen_l),
    .sram_wdata_l(sram_wdata_l),
    // act sram B interface
    .sram_rdata_b(sram_rdata_b),
    .sram_addr_b(sram_addr_b),
    .sram_wen_b(sram_wen_b),
    .sram_wdata_b(sram_wdata_b)
);



always @(*) begin
    case(state)
        IDLE: begin
            if (enable_reg) next_state = OP;
            else next_state = state;
        end
        OP: begin
            if (log_lum_done) next_state = DONE;
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
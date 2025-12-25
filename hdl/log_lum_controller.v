module log_lum_controller #(
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

    // input sram interface
    input wire [CH_NUM*BW_PER_CH-1:0] sram_rdata_in,
    output reg [20-1:0] sram_addr_in,

    // act sram L interface
    output reg [20-1:0] sram_addr_l,
    output reg sram_wen_l,
    output reg [DATA_WIDTH_L-1:0] sram_wdata_l,

    // act sram B interface
    output reg [20-1:0] sram_addr_b,
    output reg sram_wen_b,
    output reg [DATA_WIDTH_L-1:0] sram_wdata_b,

    output reg signed [20:0] b_max,
    output reg signed [20:0] b_min
);

// fsm 
localparam IDLE = 3'd0;
localparam READ = 3'd1;
localparam PIPE = 3'd2;
localparam BASE = 3'd3;
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
            if (h_counter >= 719) begin
                h_counter_next = 0;
                w_counter_next = w_counter + 1;
            end else begin
                h_counter_next = h_counter + 1;
                w_counter_next = w_counter;
            end
        end
        PIPE: begin
            if (h_counter >= 719) begin
                h_counter_next = 0;
                w_counter_next = w_counter + 1;
            end else begin
                h_counter_next = h_counter + 1;
                w_counter_next = w_counter;
            end
        end
        BASE: begin
            if (h_counter >= 719) begin
                h_counter_next = 0;
                w_counter_next = w_counter + 1;
            end else begin
                h_counter_next = h_counter + 1;
                w_counter_next = w_counter;
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
        sram_addr_in <= 0;
    end else begin
        sram_addr_in <= h_counter_next * 1280 + w_counter_next;
    end
end

reg [BW_PER_CH-1:0] R_reg;
reg [BW_PER_CH-1:0] G_reg;
reg [BW_PER_CH-1:0] B_reg;
reg [BW_PER_CH-1:0] E_reg;

always @(posedge clk) begin
    if (~srst_n) begin
        R_reg <= 0;
        G_reg <= 0;
        B_reg <= 0;
        E_reg <= 0;
    end else begin
        R_reg <= sram_rdata_in[31:24];
        G_reg <= sram_rdata_in[23:16];
        B_reg <= sram_rdata_in[15:8];
        E_reg <= sram_rdata_in[7:0];
    end
end

// log lum calculation (3 stage delay)
wire signed [20:0] log_lum_out;
log_lum log_lum_u (
    .clk(clk),
    //.srst_n(srst_n),
    .R(R_reg),
    .G(G_reg),
    .B(B_reg),
    .E(E_reg),
    .log_lum_out(log_lum_out)
);

// output sram l
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
        PIPE: begin
            if (h_out >= 719) begin
                h_out_next = 0;
                w_out_next = w_out + 1;
            end else begin
                h_out_next = h_out + 1;
                w_out_next = w_out;
            end
        end
        BASE: begin
            if (h_out >= 719) begin
                h_out_next = 0;
                w_out_next = w_out + 1;
            end else begin
                h_out_next = h_out + 1;
                w_out_next = w_out;
            end
        end
        default: begin
            h_out_next = h_out;
            w_out_next = w_out;
        end
    endcase
end
always @(posedge clk) begin
    if (~srst_n) begin
        sram_addr_l <= 0;
        sram_wen_l <= 1;
        sram_wdata_l <= 0;
    end else begin
        sram_addr_l <= h_out_next * 1280 + w_out_next;
        if (next_state == PIPE) begin
            sram_wen_l <= 0;
            sram_wdata_l <= log_lum_out;
        end else if (next_state == BASE && w_out_next <= 1279) begin
            sram_wen_l <= 0;
            sram_wdata_l <= log_lum_out;
        end else begin
            sram_wen_l <= 1;
            sram_wdata_l <= 0;
        end
    end
end

// base layer calculation
wire signed [20:0] base_layer_out;
base_layer_controller u_base_layer_controller (
    .clk(clk),
    // counter input
    .w_counter(w_out),
    .h_counter(h_out),
    // Data input
    .log_lum_out(log_lum_out),
    // Data output
    .base_layer_out(base_layer_out)
);

// output sram b
reg [10:0] w_base_out, w_base_out_next; // log2(1280) = 10.3
reg  [9:0] h_base_out, h_base_out_next; // log2(720)  = 9.4
always @(posedge clk) begin
    h_base_out <= h_base_out_next;
    w_base_out <= w_base_out_next;
end
always @(*) begin
    case(state)
        IDLE: begin
            h_base_out_next = 0;
            w_base_out_next = 0;
        end
        READ: begin
            h_base_out_next = 0;
            w_base_out_next = 0;
        end
        PIPE: begin
            h_base_out_next = 0;
            w_base_out_next = 0;
        end
        BASE: begin
            if (h_base_out >= 719) begin
                h_base_out_next = 0;
                w_base_out_next = w_base_out + 1;
            end else begin
                h_base_out_next = h_base_out + 1;
                w_base_out_next = w_base_out;
            end
        end
        default: begin
            h_base_out_next = h_base_out;
            w_base_out_next = w_base_out;
        end
    endcase
end
always @(posedge clk) begin
    if (~srst_n) begin
        sram_addr_b <= 0;
        sram_wen_b <= 1;
        sram_wdata_b <= 0;
    end else begin
        sram_addr_b <= h_base_out_next * 1280 + w_base_out_next;
        if (next_state == BASE) begin
            sram_wen_b <= 0;
            sram_wdata_b <= base_layer_out;
        end else begin
            sram_wen_b <= 1;
            sram_wdata_b <= 0;
        end
    end
end
always @(posedge clk) begin
    if (next_state == READ) begin
        b_max <= 21'sh100000;
        b_min <= 21'sh0FFFFF;
    end else if (next_state == BASE) begin
        if (base_layer_out > b_max) begin
            b_max <= base_layer_out;
        end else begin
            b_max <= b_max;
        end
        if (base_layer_out < b_min) begin
            b_min <= base_layer_out;
        end else begin
            b_min <= b_min;
        end
    end else begin
        b_max <= b_max;
        b_min <= b_min;
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
            if (h_counter >= 4) next_state = PIPE;
            else next_state = state;
        end
        PIPE: begin
            if (w_counter >= 2 && h_counter >= 17) next_state = BASE;
            else next_state = state;
        end
        BASE: begin
            if (w_base_out >= 1279 && h_base_out >= 719 ) next_state = DONE;
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
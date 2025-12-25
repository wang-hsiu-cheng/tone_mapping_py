module ldr_controller (
    input wire clk,
    input wire srst_n,
    input wire enable,
    input wire [19:0] total_pixels,
    output reg done,

    // SRAM_I Interface (Read L) -> "I" in algorithm (Q7.14)
    output reg [19:0] sram_addr_l,
    input wire signed [20:0] sram_rdata_l, 

    // SRAM_B Interface (Read B) -> "BB" in algorithm (Q7.14)
    output reg [19:0] sram_addr_b,
    input wire signed [20:0] sram_rdata_b, 

    // SRAM_IN Interface (Read Original RGBE)
    output reg [19:0] sram_addr_in,
    input wire [31:0] sram_rdata_in, // [31:24]=R, [23:16]=G, [15:8]=B, [7:0]=E

    // SRAM_OUT Interface (Write Final RGB)
    output reg [19:0] sram_addr_out,
    output reg [23:0] sram_wdata_out,
    output reg sram_wen_out
);

    //----------------------------------------------------------------
    // Parameters & State
    //----------------------------------------------------------------
    localparam S_IDLE        = 3'd0;
    localparam S_FIND_RANGE  = 3'd1;
    localparam S_LATCH_K     = 3'd2;
    localparam S_PROCESS     = 3'd3; 
    localparam S_DONE        = 3'd4;

    // Q2.15 (108853)
    localparam signed [17:0] LOG_2_10_FIXED = 18'd108853; 

    reg [2:0] state_reg, state_next;
    reg [19:0] cnt_reg, cnt_next;
    reg done_reg, done_next;

    //----------------------------------------------------------------
    // Pass 1: Min/Max Logic
    //----------------------------------------------------------------
    reg signed [20:0] max_B_reg, max_B_next;
    reg signed [20:0] min_B_reg, min_B_next;

    //----------------------------------------------------------------
    // LUT Integration (K Calculation)
    //----------------------------------------------------------------
    wire signed [20:0] b_range_wire;
    assign b_range_wire = max_B_reg - min_B_reg; // Q7.14

    // Python: divide_lut_index = np.floor(B_range / 2**8)
    wire [11:0] k_lut_addr_wire;
    assign k_lut_addr_wire = b_range_wire[19:8]; 

    wire signed [18-1:0] k_lut_data; // Q6.12
    reg signed [19-1:0] k_val_reg, k_val_next; // Q7.12 (stored)

    k_divide_lut u_k_divide_lut (
        .addr(k_lut_addr_wire), 
        .data(k_lut_data)
    );

    //----------------------------------------------------------------
    // Power LUT Integration (For Ratio Calculation)
    // Python: power_lut_index = np.floor(I_frac / 2**2)
    //----------------------------------------------------------------
    reg [13-1:0] power_lut_addr;
    wire signed [13-1:0] power_lut_data; // Q1.12
    
    power_lut u_power_lut (
        .addr(power_lut_addr),
        .data(power_lut_data)
    );

    //----------------------------------------------------------------
    // Sequential Logic
    //----------------------------------------------------------------
    always @(posedge clk) begin
        if (~srst_n) begin
            state_reg   <= S_IDLE;
            cnt_reg     <= 0;
            done_reg    <= 0;
            max_B_reg   <= -21'sd1048576;
            min_B_reg   <= 21'sd1048575;
            k_val_reg   <= 0;
        end else begin
            state_reg   <= state_next;
            cnt_reg     <= cnt_next;
            done_reg    <= done_next;
            max_B_reg   <= max_B_next;
            min_B_reg   <= min_B_next;
            k_val_reg   <= k_val_next;
        end
    end

    //----------------------------------------------------------------
    // Combinational Logic
    //----------------------------------------------------------------
    // 1. Base Layer Compression
    reg signed [20:0] comb_D;        // Q7.14
    reg signed [34:0] comb_B_mult;   // Q7.14 * Q7.12 = Q14.26
    reg signed [20:0] comb_B_comp;   // Q7.14 (Shifted >> 12)

    // 2. Reconstruction
    reg signed [20:0] comb_I_prime;  // Q7.14
    reg signed [20:0] comb_I_ratio;  // Q7.14

    // 3. Log Domain Conversion
    reg signed [38:0] comb_temp_log2_raw; // Q13.14 * Q2.15 = Q15.29
    reg signed [23:0] comb_temp_log2;     // Q9.14 (Result of >> 15)
    
    reg signed [9:0]  comb_I_int;         // Integer part
    reg [13:0]        comb_I_frac;        // Fractional part
    
    // 4. Ratio & Shift
    reg signed [13-1:0] comb_ratio;       // Q1.12
    reg signed [7:0]  e_val;              // Unsigned 8-bit treated as signed for math
    reg signed [9:0]  total_shift;        // Signed integer
    
    // 5. RGB Application
    reg [7:0] r_in, g_in, b_in;
    reg [21:0] r_temp, g_temp, b_temp; 
    
    // [MODIFIED] Increased width to 48 bits to prevent overflow during left shift
    // Python behavior implies infinite precision before clipping.
    // Max value est: 255 * 4096 (ratio) * 2^20 (shift) ~ 10^12. 
    // 48 bits covers up to ~2.8 * 10^14.
    reg [47:0] r_shifted, g_shifted, b_shifted; 
    
    reg [7:0] r_final, g_final, b_final;

    always @(*) begin
        // Defaults
        state_next  = state_reg;
        cnt_next    = cnt_reg;
        done_next   = done_reg;
        max_B_next  = max_B_reg;
        min_B_next  = min_B_reg;
        k_val_next  = k_val_reg;

        // SRAM Defaults
        sram_addr_l   = 0;
        sram_addr_b   = 0;
        sram_addr_in  = 0;
        sram_addr_out = 0;
        sram_wdata_out = 0;
        sram_wen_out   = 0;

        // LUT input default
        power_lut_addr = 0;

        // Math Intermediates Defaults
        comb_D = 0; comb_B_mult = 0; comb_B_comp = 0;
        comb_I_prime = 0; comb_I_ratio = 0;
        comb_temp_log2_raw = 0; comb_temp_log2 = 0;
        comb_I_int = 0; comb_I_frac = 0;
        comb_ratio = 0;
        e_val = 0; total_shift = 0;
        r_in = 0; g_in = 0; b_in = 0;
        r_temp = 0; g_temp = 0; b_temp = 0;
        r_shifted = 0; g_shifted = 0; b_shifted = 0;
        r_final = 0; g_final = 0; b_final = 0;

        done = done_reg;

        case (state_reg)
            S_IDLE: begin
                if (enable) begin
                    state_next = S_FIND_RANGE;
                    cnt_next   = 0;
                    max_B_next = -21'sd1048576;
                    min_B_next = 21'sd1048575;
                end
            end

            S_FIND_RANGE: begin
                // Pass 1: Scan B
                sram_addr_b = cnt_reg;
                // Latency 1 check
                if (cnt_reg > 0 && cnt_reg <= total_pixels) begin
                    if (sram_rdata_b > max_B_reg) max_B_next = sram_rdata_b;
                    if (sram_rdata_b < min_B_reg) min_B_next = sram_rdata_b;
                end
                
                if (cnt_reg < total_pixels) begin
                    cnt_next = cnt_reg + 1;
                end else begin
                    state_next = S_LATCH_K;
                    cnt_next   = 0;
                end
            end

            S_LATCH_K: begin
                // Python: k = divide_lut[...] * 2
                k_val_next = {k_lut_data, 1'b0};
                state_next = S_PROCESS;
                cnt_next   = 0;
            end

            S_PROCESS: begin
                // 1. Address Generation
                if (cnt_reg < total_pixels + 1) begin
                    cnt_next = cnt_reg + 1;
                end else begin
                    state_next = S_DONE;
                    done_next  = 1;
                end

                // SRAM Read Addresses
                if (cnt_reg < total_pixels) begin
                    sram_addr_l  = cnt_reg;
                    sram_addr_b  = cnt_reg;
                    sram_addr_in = cnt_reg; 
                end

                // 2. Data Processing Chain (Latency 1)
                if (cnt_reg >= 1 && cnt_reg <= total_pixels) begin
                    
                    // --- Step 4: Detail Layer ---
                    // D = I - BB (Q7.14)
                    comb_D = sram_rdata_l - sram_rdata_b;

                    // --- Step 5: Base Compression ---
                    // B_compressed = BB * k
                    comb_B_mult = sram_rdata_b * k_val_reg;
                    comb_B_comp = comb_B_mult[32:12]; // Matches Python floor( / 2**12)

                    // --- Step 6: Reconstruction ---
                    comb_I_prime = comb_B_comp + comb_D;
                    comb_I_ratio = comb_I_prime - sram_rdata_l;

                    // --- Log Domain Calculation ---
                    // Python: temp_log2 = trunc(I_ratio * LOG_2_10_FIXED / 2**15)
                    comb_temp_log2_raw = comb_I_ratio * LOG_2_10_FIXED;
                    comb_temp_log2 = comb_temp_log2_raw[38:15]; // Q9.14

                    // Python: I_int = floor(temp_log2 / 2**14)
                    comb_I_int = comb_temp_log2[23:14];

                    // Python: I_frac = temp_log2 - (I_int * 2**14)
                    comb_I_frac = comb_temp_log2[13:0];

                    // --- LUT Lookup ---
                    // Python: power_lut_index = floor(I_frac / 2**2)
                    power_lut_addr = comb_I_frac[13:2];
                    comb_ratio = power_lut_data; // Q1.12

                    // --- Shift Calculation ---
                    // Python: total_shift = E - 140 + I_int
                    e_val = sram_rdata_in[7:0];
                    total_shift = $signed({1'b0, e_val}) - 10'd140 + comb_I_int;

                    // --- Apply to RGB ---
                    r_in = sram_rdata_in[31:24];
                    g_in = sram_rdata_in[23:16];
                    b_in = sram_rdata_in[15:8];

                    // R_temp is roughly Q9.12
                    r_temp = r_in * $unsigned(comb_ratio);
                    g_temp = g_in * $unsigned(comb_ratio);
                    b_temp = b_in * $unsigned(comb_ratio);

                    // --- Dynamic Shift (Barrel Shifter) ---
                    if (total_shift >= 0) begin
                        // Left shift (Logical shift fills with 0s)
                        r_shifted = r_temp << total_shift;
                        g_shifted = g_temp << total_shift;
                        b_shifted = b_temp << total_shift;
                    end else begin
                        // Right shift (Logical shift >> fills with 0s)
                        // Note: -total_shift converts the negative signed value to positive magnitude
                        r_shifted = r_temp >> (-total_shift); 
                        g_shifted = g_temp >> (-total_shift);
                        b_shifted = b_temp >> (-total_shift);
                    end
                    
                    // --- Clipping / Saturation ---
                    // Python: np.clip(R_final_int, 0, 255)
                    // In hardware Q1.12 domain, "255" corresponds to (255 << 12).
                    // We check if (shifted >> 12) > 255.
                    
                    if (r_shifted[47:8] != 0) r_final = 255;
                    else r_final = r_shifted[8-1:0]; // Extract the integer part (Q12 formatted)

                    if (g_shifted[47:8] != 0) g_final = 255;
                    else g_final = g_shifted[8-1:0];

                    if (b_shifted[47:8] != 0) b_final = 255; 
                    else b_final = b_shifted[8-1:0];

                    // Write Output
                    sram_wdata_out = {r_final, g_final, b_final};
                    sram_addr_out = cnt_reg - 1;
                    sram_wen_out   = 1;
                end
            end

            S_DONE: begin
                if (!enable) state_next = S_IDLE;
            end
        endcase
    end

endmodule
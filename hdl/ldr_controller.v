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

    // [MODIFIED] SRAM_IN Interface (Read Original RGBE)
    // Python code uses "E" for shift calc. Assuming RGBE format (32-bit).
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
    // Q7.14 >> 8 = Q7.6. The integer part is used as index.
    // We assume the LUT takes the relevant bits.
    // If LUT size is small (e.g. 256), we map accordingly.
    // b_range_wire >>> 8 gives us the scaled value.
    wire [11:0] k_lut_addr_wire;
    assign k_lut_addr_wire = b_range_wire[19:8]; // Simple truncation for index

    wire signed [18-1:0] k_lut_data; // Q6.12 output from ROM
    reg signed [19-1:0] k_val_reg, k_val_next; // Store as Q7.12 (because of * 2)

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
    reg signed [20:0] comb_B_comp;   // Q13.14 (Shifted >> 12)

    // 2. Reconstruction
    reg signed [20:0] comb_I_prime;  // Q13.14 + Q7.14 = Q13.14
    reg signed [20:0] comb_I_ratio;  // Q13.14

    // 3. Log Domain Conversion
    reg signed [38:0] comb_temp_log2_raw; // Q13.14 * Q2.15 = Q15.29
    reg signed [23:0] comb_temp_log2;     // Q9.14 (Result of >> 15)
    
    reg signed [9:0]  comb_I_int;         // Integer part
    reg [13:0]        comb_I_frac;        // Fractional part (14 bits)
    
    // 4. Ratio & Shift
    reg signed [13-1:0] comb_ratio;         // Q1.12 (from LUT)
    reg signed [7:0]  e_val;              // Unsigned 8-bit, treated as pos integer
    reg signed [9:0]  total_shift;        // Signed integer
    
    // 5. RGB Application
    reg [7:0] r_in, g_in, b_in;
    reg signed [21:0] r_temp, g_temp, b_temp; // 8-bit * Q1.12 -> ~20 bits
    
    reg signed [31:0] r_shifted, g_shifted, b_shifted; // Buffer for shift result
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
                // We read k_lut_data and shift left by 1
                // Assuming k_lut_data is Q6.12, result is Q7.12
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
                    // BB(Q7.14) * k(Q7.12) = Q14.26
                    comb_B_mult = sram_rdata_b * k_val_reg;
                    
                    // Python: np.floor(B_compressed / 2**12).astype(int32) -> Q13.14
                    // Q14.26 >>> 12 = Q14.14. 
                    // We assign to 21-bit Q13.14 (assuming range fits)
                    comb_B_comp = comb_B_mult[32:12]; 

                    // --- Step 6: Reconstruction ---
                    // I_prime = B_compressed + D
                    comb_I_prime = comb_B_comp + comb_D;

                    // I_ratio = I_prime - I
                    comb_I_ratio = comb_I_prime - sram_rdata_l;

                    // --- Log Domain Calculation ---
                    // Python: temp_log2 = trunc(I_ratio * LOG_2_10_FIXED / 2**15)
                    // I_ratio(Q13.14) * LOG(Q2.15) = Q15.29
                    comb_temp_log2_raw = comb_I_ratio * LOG_2_10_FIXED;
                    
                    // Divide by 2^15 -> Shift right 15 -> Q15.14
                    // We keep meaningful bits. Python casts to int32.
                    comb_temp_log2 = comb_temp_log2_raw[38:15]; // Q9.14 (roughly)

                    // Python: I_int = floor(temp_log2 / 2**14)
                    // Extract integer part (top bits)
                    comb_I_int = comb_temp_log2[23:14];

                    // Python: I_frac = temp_log2 - (I_int * 2**14)
                    // Extract fractional part (bottom 14 bits)
                    comb_I_frac = comb_temp_log2[13:0];

                    // --- LUT Lookup ---
                    // Python: power_lut_index = floor(I_frac / 2**2)
                    // Take top 12 bits of the 14-bit fraction
                    power_lut_addr = comb_I_frac[13:2];
                    comb_ratio = power_lut_data; // Q1.12

                    // --- Shift Calculation ---
                    // Python: total_shift = E - 140 + I_int
                    // Extract E from sram_rdata_in [31:24]
                    e_val = sram_rdata_in[7:0];
                    total_shift = $signed({1'b0, e_val}) - 10'd140 + comb_I_int;

                    // --- Apply to RGB ---
                    r_in = sram_rdata_in[31:24];
                    g_in = sram_rdata_in[23:16];
                    b_in = sram_rdata_in[15:8];

                    // Python: R_temp = R * ratio
                    // 8-bit * Q1.12 = Q9.12
                    r_temp = $signed({1'b0, r_in}) * comb_ratio;
                    g_temp = $signed({1'b0, g_in}) * comb_ratio;
                    b_temp = $signed({1'b0, b_in}) * comb_ratio;

                    // --- Dynamic Shift ---
                    // Python: 
                    // if total_shift >= 0: temp << abs(shift)
                    // else:                temp >> abs(shift)
                    
                    if (total_shift >= 0) begin
                        // Left shift
                        r_shifted = r_temp << total_shift;
                        g_shifted = g_temp << total_shift;
                        b_shifted = b_temp << total_shift;
                    end else begin
                        // Right shift (using negate for abs)
                        r_shifted = r_temp >> (-total_shift);
                        g_shifted = g_temp >> (-total_shift);
                        b_shifted = b_temp >> (-total_shift);
                    end

                    // --- Final Clip & Output ---
                    // Python: np.clip(..., 0, 255)
                    // Since r_temp is Q?.12, r_shifted is also Q?.12 format logically?
                    // No, wait. Python "ratio" is from power_lut. 
                    // In Python code: R_temp = R * ratio. 
                    // Then R_final_int = R_temp << shift.
                    // The "ratio" in Python is floating point result of LUT? 
                    // Python comment: "input Q6.6 output Q6.12" for divide_lut.
                    // power_lut isn't fully spec'd in comments but implied normalized.
                    // Assuming r_shifted result needs to be integer for output.
                    // Since r_temp has 12 fractional bits (from comb_ratio Q1.12),
                    // We need to shift right by 12 to get back to integer before clipping?
                    // Python code does NOT show a division by 4096 (2^12) at the end.
                    // BUT, verify: ratio = power_lut[...]. 
                    // Usually power_lut stores 2^(fraction). Range [1.0, 2.0).
                    // If we assume the result should be integer, we must remove the fractional part of Q1.12.
                    // Let's assume we need to drop the 12 fraction bits of 'ratio' eventually.
                    // Let's look at Python: R_temp is float/double implicitly unless cast?
                    // "R_temp = R.astype(int64) * ratio". If ratio is from LUT (int), it's int.
                    // If ratio is Q1.12 integer, then R_temp is Q9.12.
                    // Then we shift. The final result R_out is uint8.
                    // This implies we need to shift right by 12 somewhere to remove the Q factor of ratio.
                    // Python code doesn't explicitly divide by 4096. 
                    // *Likely explanation*: The `total_shift` accounts for the Q-format adjustment 
                    // OR the user omitted the normalization in Python snippet.
                    // *Standard Practice*: If ratio is Q12, we must >> 12.
                    // Let's assume we need to take `r_shifted[Max:12]` as the integer.
                    
                    // Implement safe clipping on the Integer part
                    // Integer part is r_shifted >> 12
                    
                    if ((r_shifted >> 12) > 255) r_final = 255; 
                    else if ((r_shifted >> 12) < 0) r_final = 0;
                    else r_final = r_shifted[19:12]; // Extract bits 12-19

                    if ((g_shifted >> 12) > 255) g_final = 255; 
                    else if ((g_shifted >> 12) < 0) g_final = 0;
                    else g_final = g_shifted[19:12];

                    if ((b_shifted >> 12) > 255) b_final = 255; 
                    else if ((b_shifted >> 12) < 0) b_final = 0;
                    else b_final = b_shifted[19:12];

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
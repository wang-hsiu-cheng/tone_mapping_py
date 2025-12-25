`timescale 1ns/100ps

`define PAT_L 0
`define PAT_U 0 
`define NUM_PAT (`PAT_U-`PAT_L+1)

`define PAT_NAME_LENGTH 3
`define CYCLE 10
`define END_CYCLES 2000000
`define FLAG_VERBOSE 1  
`define FLAG_DUMPWV 1

module test_ltm_top;
    // Parameters
    parameter HEIGHT = 720;
    parameter WIDTH = 1280;
    parameter ADDR_WIDTH = $clog2(HEIGHT * WIDTH);
    parameter CH_NUM = 4;     
    parameter BW_PER_CH = 8;
    parameter DATA_WIDTH_L = 21;
    
    // States
    localparam LOG_LUN = 2'd0, BASE_LAYER = 2'd1, LDR = 2'd2;
    localparam L = 0, B = 1, R = 2; 

    integer test_layer;
    reg [8*26-1:0] layer_str;

    initial begin
        layer_str = 0;
        `ifdef LOG_LUN
            test_layer = LOG_LUN;
            layer_str = "      Log Luminance     ";
        `elsif BASE_LAYER
            test_layer = BASE_LAYER;
            layer_str = "      Base Layer    ";
        `elsif LDR
            test_layer = LDR;
            layer_str = "   LDR OUTPUT   ";
        `endif
    end

    integer i;
    
    reg [25*8-1:0] input_sram_golden_file;
    reg [25*8-1:0] lglum_sram_golden_file;
    reg [25*8-1:0] basel_sram_golden_file; 
    reg [25*8-1:0] outpt_sram_golden_file;

    // ===== module I/O ===== //
    reg clk;
    reg srst_n;
    reg enable;
    wire valid;

    // SRAM Wires
    wire [CH_NUM*BW_PER_CH-1:0] sram_rdata_in;
    wire [ADDR_WIDTH-1:0] sram_addr_in;
    
    wire [DATA_WIDTH_L-1:0] sram_rdata_l;
    wire [DATA_WIDTH_L-1:0] sram_wdata_l;
    wire [ADDR_WIDTH-1:0] sram_addr_l;
    wire sram_wen_l;

    wire [DATA_WIDTH_L-1:0] sram_rdata_b;
    wire [DATA_WIDTH_L-1:0] sram_wdata_b;
    wire [ADDR_WIDTH-1:0] sram_addr_b;
    wire sram_wen_b;

    // SRAM OUT Wires (Final RGB)
    wire [23:0] sram_wdata_out;
    wire [ADDR_WIDTH-1:0] sram_addr_out;
    wire sram_wen_out;

    // Instantiate LTM_top
    LTM_top #(
        .CH_NUM(CH_NUM),
        .BW_PER_CH(BW_PER_CH),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .DATA_WIDTH_L(DATA_WIDTH_L)
    ) uut (
        .clk(clk),
        .srst_n(srst_n),
        .enable(enable),
        .valid(valid),
        // sram input
        .sram_rdata_in(sram_rdata_in),
        .sram_addr_in(sram_addr_in),
        // sram act l
        .sram_rdata_l(sram_rdata_l),
        .sram_wdata_l(sram_wdata_l),
        .sram_addr_l(sram_addr_l),
        .sram_wen_l(sram_wen_l),
        // sram act b
        .sram_rdata_b(sram_rdata_b),
        .sram_wdata_b(sram_wdata_b),
        .sram_addr_b(sram_addr_b),
        .sram_wen_b(sram_wen_b),
        // sram out
        .sram_wdata_out(sram_wdata_out),
        .sram_addr_out(sram_addr_out),
        .sram_wen_out(sram_wen_out)
    );

    // ===== SRAM Models ===== //
    // SRAM INPUT
    sram_input #(
        .CH_NUM(CH_NUM),
        .BW_PER_CH(BW_PER_CH),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH)
    ) sram_input_u (
        .clk(clk),
        .csb(1'b0),
        .wsb(1'b1),
        .wdata({CH_NUM*BW_PER_CH{1'b0}}),
        .waddr(sram_addr_in), 
        .raddr(sram_addr_in), 
        .rdata(sram_rdata_in)
    );

    // SRAM ACT L
    sram_act_l #(
        .DATA_WIDTH(DATA_WIDTH_L),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH)
    ) sram_act_l_u (
        .clk(clk),
        .csb(1'b0),
        .wsb(sram_wen_l),
        .wdata(sram_wdata_l), 
        .waddr(sram_addr_l), 
        .raddr(sram_addr_l), 
        .rdata(sram_rdata_l)
    );

    // SRAM ACT B
    sram_act_b #(
        .DATA_WIDTH(DATA_WIDTH_L),
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH)
    ) sram_act_b_u (
        .clk(clk),
        .csb(1'b0),
        .wsb(sram_wen_b),
        .wdata(sram_wdata_b), 
        .waddr(sram_addr_b), 
        .raddr(sram_addr_b), 
        .rdata(sram_rdata_b)
    );

    // SRAM OUTPUT (Stores RTL Result)
    sram_output #(
        .CH_NUM(3),        // RGB 3 channels
        .BW_PER_CH(8),     // 8 bits per channel
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH)
    ) sram_output_u (
        .clk(clk),
        .csb(1'b0),        
        .wsb(sram_wen_out), // Active Low write enable
        .wdata(sram_wdata_out),
        .waddr(sram_addr_out),
        .raddr(sram_addr_out), 
        .rdata()           
    );

    // ===== Waveform ===== //
    `define SDFFILE "../syn/netlist/LTM_top_syn.sdf"
    `ifdef SDF
        initial $sdf_annotate(`SDFFILE, uut);
    `endif
    initial begin
        if(`FLAG_DUMPWV)begin
            // $fsdbDumpfile("test_ltm_ldr.fsdb");
            $fsdbDumpvars(1, LTM_top);
            $fsdbDumpvars(1, log_lum_controller);
            $fsdbDumpvars(1, ldr_controller);
        end
    end

    // ===== Golden Data Storage ===== //
    reg [31:0] input_sram_value [0:HEIGHT*WIDTH-1];
    reg [31:0] lglum_sram_value [0:HEIGHT*WIDTH-1]; 
    reg [31:0] basel_sram_value [0:HEIGHT*WIDTH-1];
    reg [23:0] outpt_sram_value [0:HEIGHT*WIDTH-1];

    // ===== Clock Gen ===== //
    initial begin
        clk = 0;
        while(1) #(`CYCLE/2) clk = ~clk;
    end

    // ===== Timeout ===== //
    initial begin
        #(`CYCLE * `END_CYCLES);
        $display("Error!!! Simulation Timeout.");
        $finish;
    end

    // ===== cycle counter ===== //
integer cycle_cnt;
integer aver_cycle_cnt;
initial begin
    cycle_cnt = 0;
    aver_cycle_cnt = 0;
    while(1) begin 
        cycle_cnt = cycle_cnt + 1;
        @(negedge clk);
    end
end

// ===== output comparision ===== //
integer error_total;
integer error_tmp;
integer pat_idx;
integer total_err_pat;

initial begin
    // check if PAT_L and PAT_U are both valid
    if((`PAT_L < 0) || (`PAT_L > 60-1) || (`PAT_U < 0) || (`PAT_U > 60-1)) begin
        $display("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        $display("X                                                                             X");
        $display("X   Error!!! PAT_L and PAT_U should be within the range [0, %3d]              X", 60-1);
        $display("X                                                                             X");
        $display("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        $finish;
    end
    else if(`PAT_L > `PAT_U) begin
        $display("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        $display("X                                                        X");
        $display("X   Error!!! PAT_L should be smaller or equal to PAT_U   X");
        $display("X                                                        X");
        $display("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        $finish;    
    end

    // show simulation configuration
    $display("\n%c[1;36mStart checking %s layer ... %c[0m\n", 27, layer_str, 27);

    total_err_pat = 0;
    for (pat_idx=`PAT_L; pat_idx<=`PAT_U; pat_idx=pat_idx+1) begin
        
        // reset sram
        sram_input_u.reset_sram;
        sram_act_l_u.reset_sram;
        sram_act_b_u.reset_sram;
        sram_output_u.reset_sram;

        error_total = 0;

        // load golden for this test number
        load_golden(pat_idx, test_layer);

        $display("\n========================================================================");
        $display("======================== Pattern No. %02d ========================", pat_idx);
        $display("========================================================================");
        $display();

        srst_n = 1;
        enable = 0;
        @(negedge clk); srst_n = 1'b0;
        @(negedge clk); srst_n = 1'b1; enable = 1'b1;
        @(negedge clk); enable = 1'b0;

        wait(valid);

        @(negedge clk);
        case (test_layer)
            LOG_LUN:    compare_output(L);
            BASE_LAYER: compare_output(B);
            LDR:        compare_output(R);
        endcase
    end
    
    aver_cycle_cnt = cycle_cnt / `NUM_PAT;

    // summary of all pattern
    $display("\n\n\n                   Summary of all pattern: ");
    if(total_err_pat == 0) begin 
        $display("------------------------------------------------------------\n");
        $write("%c[1;32mCongratulations! %c[0m",27, 27);
        $display("Your %s layer is correct!", layer_str);
        $display("Total cycle count = %0d", cycle_cnt);
        $display("Average cycle count per pattern = %0d", aver_cycle_cnt);
        $display("-----------------------------PASS---------------------------\n");
        
    end else begin
        $display("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        $display("X                                                            X");
        $display("X        %c[1;31mFAIL%c[0m in %-26s layer!!!         X",27,27, layer_str);
        $display("X              %3d patterns are failed... (T ~ T)            X", total_err_pat);
        $display("X                                                            X");
        $display("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        $display("Total cycle count = %0d", cycle_cnt);
        $display("Average cycle count per pattern = %0d", aver_cycle_cnt);
    end
    $finish;
end


task load_golden(
    input integer index,
    input integer layer
);
    reg [8-1:0] index_digit_2, index_digit_1, index_digit_0;
    begin
        // Decode file name index bit
        index_digit_2 = (index % 1000) / 100 + 48;
        index_digit_1 = (index % 100 ) / 10 + 48;
        index_digit_0 = (index % 10  ) + 48;

        // Dat file name
        input_sram_golden_file = "param/input/input_000.dat"; // 25 char
        lglum_sram_golden_file = "param/lglum/lglum_000.dat"; // 25 char
        basel_sram_golden_file = "param/basel/basel_000.dat"; // 25 char
        outpt_sram_golden_file = "param/outpt/outpt_000.dat"; // 25 char

        // Change dat test number
        input_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] 
            = {index_digit_2, index_digit_1, index_digit_0};
        lglum_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] 
            = {index_digit_2, index_digit_1, index_digit_0};
        basel_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] 
            = {index_digit_2, index_digit_1, index_digit_0};
        outpt_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] 
            = {index_digit_2, index_digit_1, index_digit_0};

        // load input
        $readmemh(input_sram_golden_file, input_sram_value);

        // store input data into sram
        for(i=0; i<HEIGHT*WIDTH; i=i+1) begin
            sram_input_u.load_act(i, input_sram_value[i]);
        end
    end
endtask

reg [7:0] hw_R, hw_G, hw_B;
reg [7:0] golden_R, golden_G, golden_B;

task compare_output(input integer sram_sel);
    integer h, w;
    integer error_tmp;
    case (sram_sel)
        L: begin
            // load golden
            $readmemh(lglum_sram_golden_file, lglum_sram_value);

            for(h=0; h<HEIGHT; h=h+1) begin
                error_tmp = 0;
                for(w=0; w<WIDTH; w=w+1) begin
                    if((lglum_sram_value[h*WIDTH+w][20:0] !== sram_act_l_u.mem[h*WIDTH+w][20:0]))
                        error_tmp = error_tmp + 1;
                end
                if (error_tmp != 0) begin
                    if(`FLAG_VERBOSE) $display("Sram #L row %0d FAIL!", (h));
                    error_total = error_total + 1;
                end else begin
                    if(`FLAG_VERBOSE) $display("Sram #L row %0d PASS!", (h));
                end
            end
            // summary of this pattern
            if(`FLAG_VERBOSE) $display("\n========================================================================");
            if(error_total == 0) begin
                if(`FLAG_VERBOSE) $display("Congratulations! Your %s layer is correct!", layer_str);
                if(`FLAG_VERBOSE) $display("Pattern No. %02d is successfully passed !", pat_idx);
                else              $write("%c[1;32mPASS! %c[0m",27, 27);
            end else begin
                if(`FLAG_VERBOSE) $display("There are total %0d row hase errors in your %s layer.", error_total, layer_str);
                if(`FLAG_VERBOSE) $display("Pattern No. %02d is failed...", pat_idx);
                else              $write("%c[1;31mFAIL! %c[0m",27, 27);
                total_err_pat = total_err_pat + 1;
            end
            if(`FLAG_VERBOSE) $display("========================================================================");
            // $finish;
        end

        B: begin
            // load golden
            $readmemh(basel_sram_golden_file, basel_sram_value);
            
            for(h=0; h<HEIGHT; h=h+1) begin
                error_tmp = 0;
                for(w=0; w<WIDTH; w=w+1) begin
                    if((basel_sram_value[h*WIDTH+w][20:0] !== sram_act_b_u.mem[h*WIDTH+w][20:0]))
                        error_tmp = error_tmp + 1;
                end
                if (error_tmp != 0) begin
                    if(`FLAG_VERBOSE) $display("Sram #B row %0d FAIL!", (h));
                    error_total = error_total + 1;
                end else begin
                    if(`FLAG_VERBOSE) $display("Sram #B row %0d PASS!", (h));
                end
            end
            // summary of this pattern
            if(`FLAG_VERBOSE) $display("\n========================================================================");
            if(error_total == 0) begin
                if(`FLAG_VERBOSE) $display("Congratulations! Your %s layer is correct!", layer_str);
                if(`FLAG_VERBOSE) $display("Pattern No. %02d is successfully passed !", pat_idx);
                else              $write("%c[1;32mPASS! %c[0m",27, 27);
            end else begin
                if(`FLAG_VERBOSE) $display("There are total %0d row hase errors in your %s layer.", error_total, layer_str);
                if(`FLAG_VERBOSE) $display("Pattern No. %02d is failed...", pat_idx);
                else              $write("%c[1;31mFAIL! %c[0m",27, 27);
                total_err_pat = total_err_pat + 1;
            end
            if(`FLAG_VERBOSE) $display("========================================================================");
            // $finish;
        end

        R: begin
            // load golden
            $readmemh(outpt_sram_golden_file, outpt_sram_value);

            for(h=0; h<HEIGHT; h=h+1) begin
                error_tmp = 0;
                for(w=0; w<WIDTH; w=w+1) begin
                    if((outpt_sram_value[h*WIDTH+w][23:0] !== sram_output_u.mem[h*WIDTH+w][23:0])) begin
                        error_tmp = error_tmp + 1;
                    
                        hw_R = sram_output_u.mem[h*WIDTH+w][23:16];
                        hw_G = sram_output_u.mem[h*WIDTH+w][15:8];
                        hw_B = sram_output_u.mem[h*WIDTH+w][7:0];
                        golden_R = outpt_sram_value[h*WIDTH+w][23:16];
                        golden_G = outpt_sram_value[h*WIDTH+w][15:8];
                        golden_B = outpt_sram_value[h*WIDTH+w][7:0];
                        if(`FLAG_VERBOSE) $display("Error at [%0d,%0d]: HW_R=%d, HW_G=%d, HW_B=%d, G_R=%d, G_G=%d, G_B=%d", h, w, hw_R, hw_G, hw_B, golden_R, golden_G, golden_B);
                    end
                end
                if (error_tmp != 0) begin
                    if(`FLAG_VERBOSE) $display("Sram #O row %0d FAIL!", (h));
                    error_total = error_total + 1;
                end else begin
                    if(`FLAG_VERBOSE) $display("Sram #O row %0d PASS!", (h));
                end
            end
            // summary of this pattern
            if(`FLAG_VERBOSE) $display("\n========================================================================");
            if(error_total == 0) begin
                if(`FLAG_VERBOSE) $display("Congratulations! Your %s layer is correct!", layer_str);
                if(`FLAG_VERBOSE) $display("Pattern No. %02d is successfully passed !", pat_idx);
                else              $write("%c[1;32mPASS! %c[0m",27, 27);
            end else begin
                if(`FLAG_VERBOSE) $display("There are total %0d row hase errors in your %s layer.", error_total, layer_str);
                if(`FLAG_VERBOSE) $display("Pattern No. %02d is failed...", pat_idx);
                else              $write("%c[1;31mFAIL! %c[0m",27, 27);
                total_err_pat = total_err_pat + 1;
            end
            if(`FLAG_VERBOSE) $display("========================================================================");
            // $finish;
        end

    endcase
endtask

endmodule

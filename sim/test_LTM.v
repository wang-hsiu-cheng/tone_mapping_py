`timescale 1ns/100ps

`define PAT_L 0
`define PAT_U 1
`define NUM_PAT (`PAT_U-`PAT_L+1)

`define PAT_NAME_LENGTH 3
`define CYCLE 10
`define END_CYCLES 2000000
`define FLAG_VERBOSE 1  
`define FLAG_DUMPWV 1

module test_ltm_top;

// Parameters (adjust as needed)
parameter HEIGHT = 720;
parameter WIDTH = 1280;
parameter ADDR_WIDTH = $clog2(HEIGHT * WIDTH);
// For input sram data per channel = 32 bits
parameter CH_NUM = 4;     // RGBE 4 channel
parameter BW_PER_CH = 8;  // each channel has 8 bits
// For sram act L & B data per channel = 21 bits (Q7.14)
parameter DATA_WIDTH_L = 21;


localparam LOG_LUN = 2'd0, BASE_LAYER = 2'd1, LDR = 2'd2;
localparam L = 0, B = 1;

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
// ===== pattern files ===== //
reg [25*8-1:0] input_sram_golden_file; 
reg [25*8-1:0] lglum_sram_golden_file;
reg [25*8-1:0] basel_sram_golden_file; 

// ===== module I/O ===== //
reg clk;
reg srst_n;
reg enable;
wire valid;
// SRAM input connection
wire [CH_NUM*BW_PER_CH-1:0] sram_rdata_in;
wire [ADDR_WIDTH-1:0] sram_addr_in;
// SRAN ACT L connection
wire [DATA_WIDTH_L-1:0] sram_rdata_l;
wire [DATA_WIDTH_L-1:0] sram_wdata_l;
wire [ADDR_WIDTH-1:0] sram_addr_l;
wire sram_wen_l;
// SRAN ACT B connection
wire [DATA_WIDTH_L-1:0] sram_rdata_b;
wire [DATA_WIDTH_L-1:0] sram_wdata_b;
wire [ADDR_WIDTH-1:0] sram_addr_b;
wire sram_wen_b;


// Instantiate ViT RTL module
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
// sram act l
.sram_rdata_b(sram_rdata_b),
.sram_wdata_b(sram_wdata_b),
.sram_addr_b(sram_addr_b),
.sram_wen_b(sram_wen_b)
);

// ===== sram connection ===== //
// SRAM for INPUT
sram_input #(
    .CH_NUM(CH_NUM),
    .BW_PER_CH(BW_PER_CH),
    .HEIGHT(HEIGHT),
    .WIDTH(WIDTH)
) sram_input_u (
    .clk(clk),
    .csb(1'b0),
    .wsb(1'b1), // no use
    .wdata({CH_NUM*BW_PER_CH{1'b0}}), // no use
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

// ===== waveform dumpping ===== //
initial begin
    if(`FLAG_DUMPWV)begin
        $fsdbDumpfile("fp_local_tone_mapping.fsdb");
        $fsdbDumpvars(0, LTM_top);
    end
end

// ===== parameter & golden answers ===== //
reg [CH_NUM*BW_PER_CH-1:0] input_sram_value [0:HEIGHT*WIDTH-1];
reg [32-1:0] lglum_sram_value [0:HEIGHT*WIDTH-1]; // 32 bit for saving .dat but [20:0] is golden
reg [32-1:0] basel_sram_value [0:HEIGHT*WIDTH-1]; // 32 bit for saving .dat but [20:0] is golden

// ===== system reset ===== //
initial begin
    clk = 0;
    while(1) #(`CYCLE/2) clk = ~clk;
end

initial begin
  #(`CYCLE * `END_CYCLES);
    $display("\n========================================================");
    $display("   Error!!! Simulation time is too long...            ");
    $display("   There might be something wrong in your code.       ");
    $display("   If your design really needs such a long time,      ");
    $display("   increase the END_CYCLES setting in the testbench.  ");
    $display("========================================================");
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
    if((`PAT_L < 0) || (`PAT_L > `NUM_PAT-1) || (`PAT_U < 0) || (`PAT_U > `NUM_PAT-1)) begin
        $display("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        $display("X                                                                             X");
        $display("X   Error!!! PAT_L and PAT_U should be within the range [0, %3d]              X", `NUM_PAT-1);
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
        
        // Change dat test number
        input_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] 
            = {index_digit_2, index_digit_1, index_digit_0};
        lglum_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] 
            = {index_digit_2, index_digit_1, index_digit_0};
        basel_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] 
            = {index_digit_2, index_digit_1, index_digit_0};

        // load input
        $readmemh(input_sram_golden_file, input_sram_value);

        // load golden
        $readmemh(lglum_sram_golden_file, lglum_sram_value);
        $readmemh(basel_sram_golden_file, basel_sram_value);

        // store input data into sram
        for(i=0; i<HEIGHT*WIDTH; i=i+1) begin
            sram_input_u.load_act(i, input_sram_value[i]);
        end
    end
endtask

task compare_output(input integer sram_sel);
    integer h, w;
    integer error_tmp;
    case (sram_sel)
        L: begin
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

    endcase
endtask

endmodule
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

    reg [8*26-1:0] layer_str;

    initial begin
        layer_str = "   LDR OUTPUT   ";
    end

    integer i;
    
    reg [25*8-1:0] input_sram_golden_file;
    reg [25*8-1:0] lglum_sram_golden_file;
    reg [25*8-1:0] basel_sram_golden_file; 
    reg [27*8-1:0] ldr_sram_golden_file;

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
    ) sram_out_u (
        .clk(clk),
        .csb(1'b0),        
        .wsb(~sram_wen_out), // Active Low write enable
        .wdata(sram_wdata_out),
        .waddr(sram_addr_out),
        .raddr(sram_addr_out), 
        .rdata()           
    );

    // ===== Waveform ===== //
    initial begin
        if(`FLAG_DUMPWV)begin
            $fsdbDumpfile("test_ltm_ldr.fsdb");
            $fsdbDumpvars(0, test_ltm_top);
        end
    end

    // ===== Golden Data Storage ===== //
    reg [31:0] input_sram_value [0:HEIGHT*WIDTH-1];
    reg [31:0] lglum_sram_value [0:HEIGHT*WIDTH-1]; 
    reg [31:0] basel_sram_value [0:HEIGHT*WIDTH-1];
    reg [23:0] ldr_sram_value   [0:HEIGHT*WIDTH-1]; 

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

    // ===== Main Testing Flow ===== //
    integer cycle_cnt;
    integer pat_idx;
    integer error_total;

    initial begin
        cycle_cnt = 0;

        $display("\n%c[1;36mStart LDR Verification (Comparing with Golden) ... %c[0m\n", 27, 27);

        for (pat_idx=`PAT_L; pat_idx<=`PAT_U; pat_idx=pat_idx+1) begin
            
            // 1. Reset SRAMs
            sram_input_u.reset_sram;
            sram_act_l_u.reset_sram;
            sram_act_b_u.reset_sram;
            sram_out_u.reset_sram; 

            // 2. Load Input & ALL Golden Data (including Output)
            load_all_data(pat_idx); 
            
            // [INJECT] Pre-load L and B SRAMs with Golden Data
            for(i=0; i<HEIGHT*WIDTH; i=i+1) begin
                sram_act_l_u.load_act(i, lglum_sram_value[i][20:0]);
                sram_act_b_u.load_act(i, basel_sram_value[i][20:0]);
            end

            $display("Processing Pattern No. %02d", pat_idx);

            // 3. Start Simulation
            srst_n = 1;
            enable = 0;
            @(negedge clk); srst_n = 1'b0;
            @(negedge clk); srst_n = 1'b1; enable = 1'b1;
            @(negedge clk); enable = 1'b0;

            // 4. [HACK] Wait for OP state to finish, then RE-FORCE SRAM content
            wait(uut.log_lum_done); 
            
            $display(">> Log Lum Done. Re-injecting Golden I/B...");
            
            for(i=0; i<HEIGHT*WIDTH; i=i+1) begin
                sram_act_l_u.mem[i] = lglum_sram_value[i][20:0];
                sram_act_b_u.mem[i] = basel_sram_value[i][20:0];
            end

            // 5. Wait for LDR to finish
            wait(valid);
            @(negedge clk);

            // 6. Compare Output with Golden
            compare_output(pat_idx);
            
        end
        
        $display("\nAll patterns processed.");
        $finish;
    end

    // Cycle Counter
    always @(posedge clk) cycle_cnt = cycle_cnt + 1;

    // Load Data Task
    task load_all_data(input integer index);
        reg [8-1:0] d2, d1, d0;
        begin
            d2 = (index % 1000) / 100 + 48;
            d1 = (index % 100 ) / 10 + 48;
            d0 = (index % 10  ) + 48;

            input_sram_golden_file = "param/input/input_000.dat";
            lglum_sram_golden_file = "param/lglum/lglum_000.dat";
            basel_sram_golden_file = "param/basel/basel_000.dat";
            ldr_sram_golden_file   = "param/output/output_000.dat"; 

            // [FIXED] 使用 4*8 位移量，對應 _000.dat 的三個數字位置
            input_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] = {d2, d1, d0};
            lglum_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] = {d2, d1, d0};
            basel_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8] = {d2, d1, d0};
            ldr_sram_golden_file[4*8+:`PAT_NAME_LENGTH*8]   = {d2, d1, d0}; // 修正這裡！

            $readmemh(input_sram_golden_file, input_sram_value);
            $readmemh(lglum_sram_golden_file, lglum_sram_value);
            $readmemh(basel_sram_golden_file, basel_sram_value);
            $readmemh(ldr_sram_golden_file,   ldr_sram_value);   

            // Load Input SRAM
            for(i=0; i<HEIGHT*WIDTH; i=i+1) begin
                sram_input_u.load_act(i, input_sram_value[i]);
            end
        end
    endtask

    // Comparison Task
    task compare_output(input integer index);
        integer k;
        integer err_cnt;
        reg [23:0] golden;
        reg [23:0] rtl;
        begin
            err_cnt = 0;
            for (k = 0; k < HEIGHT * WIDTH; k = k + 1) begin
                golden = ldr_sram_value[k];
                rtl    = sram_out_u.mem[k][23:0]; 

                if (golden !== rtl) begin
                    err_cnt = err_cnt + 1;
                    if (`FLAG_VERBOSE && err_cnt <= 10) begin 
                        $display("Error at pixel %0d: Golden=%h, RTL=%h", k, golden, rtl);
                    end
                end
            end

            if (err_cnt == 0) begin
                $display("\n%c[1;32m[PASS] Pattern %02d Matches Golden Output!%c[0m", 27, index, 27);
            end else begin
                $display("\n%c[1;31m[FAIL] Pattern %02d has %0d errors.%c[0m", 27, index, err_cnt, 27);
            end
        end
    endtask

endmodule

func.func @ssa_is_fun() -> i32 { //ctional programming
    // i = 1
    // j = 1
    // k = 0
    %i = arith.constant 1 : i32
    %j = arith.constant 1 : i32
    %k = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c20 = arith.constant 20 : i32
    %c100 = arith.constant 100 : i32
    // while k < 100
    %k_2, %j_2 = scf.while (%k_1 = %k, %j_1 = %j) : (i32, i32) -> (i32, i32) {
        %condition = arith.cmpi slt, %k_1, %c100 : i32
        scf.condition(%condition) %k_1, %j_1 : i32, i32
    } do {
    ^bb0(%k_2: i32, %j_2: i32):
        // if j < 20
        %cond_j = arith.cmpi slt, %j_2, %c20 : i32
        %k_3, %j_3 = scf.if %cond_j -> (i32, i32) {
            // j = i
            // k = k + 1
            %k_3 = arith.addi %k_2, %c1 : i32
            scf.yield %k_3, %i : i32, i32
        } else {
            // j = k
            // k = k + 2
            %k_3 = arith.addi %k_2, %c2 : i32
            scf.yield %k_3, %k_2 : i32, i32
        }
        scf.yield %k_3, %j_3 : i32, i32
    }
    return %j : i32
}

// Notes:
// - In the different regions the ks can have similar names because of scoping introduced by regions
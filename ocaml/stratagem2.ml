(*********************************************************************)
(*                                                                   *)
(*                      STRATAGEM Version 2                          *)
(*                      Complete source code                         *)
(*                                                                   *)
(*  (C) Daniel Hillerström &                                         *)
(*      John Longley, The University of Edinburgh, March2023         *)
(*                      Version 2: March 2023                        *)
(*                                                                   *)
(*********************************************************************)
(*                                                                   *)
(* Complete OCaml source code for the STRATAGEM system, Version 2.   *)
(* Contains signature declarations, functor bodies, build commands.  *)
(* OCaml 5.00.                                                       *)
(* See the accompanying file "userdoc.txt" for background and        *)
(* user documentation.                                               *)
(*                                                                   *)
(*********************************************************************)

module Move: sig
  type t = Basic of int
         | Pair of t * t
         | Question of t
         | Answer of t

  val basic : int -> t
  val pair : t -> t -> t
  val question : t -> t
  val answer : t -> t
end = struct
  type t = Basic of int
         | Pair of t * t
         | Question of t
         | Answer of t

  let basic n = Basic n
  let pair m m' = Pair (m, m')
  let question m = Question m
  let answer m = Answer m
end

module Forest: sig
  type t = Forest of (Move.t -> Move.t * t)

  val (%) : t -> Move.t -> Move.t * t
  val run : t -> Move.t -> Move.t * t

  val apply : t -> t -> t
  val lambda : (t -> t) -> t

  module Object: sig
    type forest = t
    type t = Move.t -> Move.t
    val of_forest : forest -> t
    val to_forest : t -> forest
  end
end = struct
  type t = Forest of (Move.t -> Move.t * t)

  let run : t -> Move.t -> Move.t * t
    = fun (Forest f) i -> f i
  let (%) : t -> Move.t -> Move.t * t
    = run
  let (%%) = (%)

  let rec apply : t -> t -> t
    = fun f g ->
    let rec apply' r g =
      match r with
      | (Move.Question m, f) ->
         (m, apply f g)
      | (Move.Answer m, f) ->
         let (j,g') = g % m in
         apply' (f %% j) g'
      | _ -> assert false
    in
    Forest (fun i -> apply' (f %% i) g)


  module Lambda = struct
    type _ Effect.t += Branch : Move.t -> (Move.t * t) Effect.t
    type branch = Move.t -> Move.t * t

    let branch q =
      Effect.perform (Branch (Move.Question q))

    let rec hbranch =
      let open Effect.Deep in
      { retc = (fun (ans, p') -> (Move.Answer ans, lambda' (fun _ -> p'))) (* TODO it should be possible to drop the dummy abstraction. *)
      ; exnc = raise
      ; effc = (fun (type a) (eff : a Effect.t) ->
        match eff with
        | Branch q ->
           Some
             (fun (k : (a, _) continuation) ->
               let open Multicont.Deep in
               let r = promote k in
               (q, Forest
                     (fun (a : Move.t) ->
                       lambda''
                         (fun (br : branch) ->
                           resume r (a, Forest br)))))
        | _ -> None) }
    and lambda phi = lambda' (fun f -> phi (Forest f))
    and lambda' : (branch -> t) -> t
      = fun p ->
      Forest
        (fun h ->
          lambda''
            (fun g ->
              let f = p g in
              f % h))
    and lambda'' : (branch -> Move.t * t) -> Move.t * t
      = fun r ->
      Effect.Deep.match_with r branch hbranch
  end

  let lambda : (t -> t) -> t
    = Lambda.lambda

  module Object = struct
    type forest = t
    type t = Move.t -> Move.t

    let of_forest : forest -> t
      = fun f ->
      let fstore = ref f in
      fun i -> let (n, f') = !fstore % i in
               fstore := f'; i

    let rec to_forest : t -> forest
      = fun o ->
      Forest
        (fun i ->
          let m = o i in
          (m, to_forest o))
  end
end

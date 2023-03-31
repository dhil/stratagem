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
  type t = Atom of int
         | Pair of t * t
         | Inl of t
         | Inr of t
         | Question of t
         | Answer of t

  val atom : int -> t
  val pair : t -> t -> t
  val inl : t -> t
  val inr : t -> t

  val question : t -> t
  val answer : t -> t
end = struct
  type t = Atom of int
         | Pair of t * t
         | Inl of t
         | Inr of t
         | Question of t
         | Answer of t

  let atom n = Atom n
  let pair m m' = Pair (m, m')
  let inl m = Inl m
  let inr m = Inr m

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
  let (%%) : t -> Move.t -> Move.t * t
    = (%)

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

(* The next task is to implement the retraction (!forest) <| forest *)
module Retraction = struct

  let rec decode_left : Move.t -> int = function
    | Move.(Pair (Atom i, _)) -> i
    | Move.Pair (m, _) -> decode_left m
    | _ -> assert false

  let embed : Forest.t -> Forest.t
    = fun f ->
    let rec embed' : Forest.t list -> (Move.t -> Move.t * Forest.t)
      = fun subforests h ->
      match h with
      | Move.Pair (i, j) ->
         let open Forest in
         let i' = decode_left i in
         let f = List.nth subforests i' in
         let (k, g) = f % j in
         (k, Forest (embed' (subforests @ [g])))
      | _ -> assert false
    in
    Forest (embed' [f])

  let project : Forest.t -> Forest.t
    = fun f ->
    let rec project_object : Forest.Object.t -> Forest.t
      = fun o ->
      let counter = ref 0 in
      let rec copy i j =
        let m = o (Move.Pair (i, j)) in
        (m, Forest.Forest (copy (incr counter; Move.Atom !counter)))
      in Forest.Forest (copy (Move.Atom 0))
    in
    project_object (Forest.Object.of_forest f)
end

let rec diverge _ =
  let exception Diverge in
  raise Diverge

let divergent_forest = Forest.Forest diverge

let atom2atom_embed : (Move.t -> Move.t) -> Forest.t
  = fun h ->
  Forest.Forest (fun i -> (h i, divergent_forest))

let atom2atom_project : Forest.t -> Move.t -> Move.t
  = fun f i -> fst Forest.(f % i)

let atom2forest_embed : (Move.t -> Forest.t) -> Forest.t
  = fun i -> Forest.Forest (fun j -> (Move.Atom 0, i j))

let atom2forest_project : Forest.t -> Move.t -> Forest.t
  = fun f i -> snd Forest.(f % i)

let example () =
  let open Move in
  let open Forest in
  let lam =
    lambda
      (fun f ->
        Forest (fun i -> (atom 42, f)))
  in
  apply lam (Forest (fun i -> (atom 32, divergent_forest)))

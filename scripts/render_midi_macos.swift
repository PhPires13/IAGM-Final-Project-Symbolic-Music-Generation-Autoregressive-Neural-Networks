// Renderiza um MIDI em WAV no macOS, sem dependências externas: usa o AVAudioEngine em modo
// offline com o banco de sons GS que já vem no sistema (piano acústico, programa 0).
// Alternativa ao FluidSynth para quem não tem um soundfont instalado.
//
// Uso: swift scripts/render_midi_macos.swift entrada.mid saida.wav
import AVFoundation

let args = CommandLine.arguments
guard args.count == 3 else {
    FileHandle.standardError.write("uso: render_midi_macos.swift entrada.mid saida.wav\n".data(using: .utf8)!)
    exit(2)
}
let midiURL = URL(fileURLWithPath: args[1])
let wavURL = URL(fileURLWithPath: args[2])
let soundBank = URL(fileURLWithPath: "/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls")

let engine = AVAudioEngine()
let sampler = AVAudioUnitSampler()
engine.attach(sampler)
engine.connect(sampler, to: engine.mainMixerNode, format: nil)

let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 2)!
let frames: AVAudioFrameCount = 4096
try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: frames)
try sampler.loadSoundBankInstrument(at: soundBank, program: 0, bankMSB: 0x79, bankLSB: 0)

let sequencer = AVAudioSequencer(audioEngine: engine)
try sequencer.load(from: midiURL, options: [])
for track in sequencer.tracks { track.destinationAudioUnit = sampler }
let seconds = sequencer.tracks.map { $0.lengthInSeconds }.max() ?? 0
let tail = 2.0  // deixa as notas finais soarem

try engine.start()
sequencer.prepareToPlay()
try sequencer.start()

// WAV de 16 bits; o arquivo é fechado (e o cabeçalho escrito) ao sair da função
func renderToFile() throws {
    let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: format.sampleRate,
        AVNumberOfChannelsKey: 2, AVLinearPCMBitDepthKey: 16,
        AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
    ]
    let out = try AVAudioFile(forWriting: wavURL, settings: settings,
                              commonFormat: .pcmFormatFloat32, interleaved: false)
    let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: frames)!
    let total = AVAudioFramePosition((seconds + tail) * format.sampleRate)
    while engine.manualRenderingSampleTime < total {
        let n = min(frames, AVAudioFrameCount(total - engine.manualRenderingSampleTime))
        let status = try engine.renderOffline(n, to: buffer)
        guard status == .success else { break }
        try out.write(from: buffer)
    }
}
try renderToFile()
sequencer.stop()
engine.stop()

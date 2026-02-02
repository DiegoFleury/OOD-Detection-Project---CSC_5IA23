"""
Neural Collapse Analysis Module
===============================

Análise completa de Neural Collapse ao longo do treinamento usando checkpoints.

Baseado em:
- Papyan et al. "Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training" (PNAS 2020)
- Han et al. "Neural Collapse Under MSE Loss" (ICLR 2022)

Métricas implementadas:
- NC1: Activation Collapse (tr{Sw @ Sb^-1} → 0)
- NC2: Equinorm (CoV das normas → 0)
- NC2: Equiangularity (mutual coherence → 0)
- NC3: Self-Duality (||W^T - M||^2 → 0)
- NC4: Convergence to NCC (mismatch → 0)

Uso:
    from nc_analysis import load_checkpoints_and_analyze, plot_nc_evolution
    
    tracker = load_checkpoints_and_analyze(
        checkpoint_dir='path/to/checkpoints',
        model_class=ResNet18,
        loader=train_loader,
        device='cuda',
        num_classes=100
    )
    
    plot_nc_evolution(tracker, save_dir='path/to/figures')
"""

import os
import re
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse.linalg import svds
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable


# ==============================================================================
# 1. Container de métricas
# ==============================================================================

@dataclass
class NCMetricsTracker:
    """
    Container para armazenar métricas de Neural Collapse ao longo do treinamento.
    
    Atributos:
        epochs: Lista de epochs analisadas
        accuracy: Acurácia do modelo
        loss: Loss (CrossEntropy ou MSE)
        Sw_invSb: NC1 - tr{Sw @ Sb^-1}
        norm_M_CoV: NC2 - Coeficiente de variação das normas das médias de classe
        norm_W_CoV: NC2 - Coeficiente de variação das normas dos classificadores
        cos_M: NC2 - Coerência mútua das médias de classe
        cos_W: NC2 - Coerência mútua dos pesos do classificador
        W_M_dist: NC3 - ||W^T - M||^2 (normalizado)
        NCC_mismatch: NC4 - Proporção de mismatch entre predição e NCC
    """
    epochs: List[int] = field(default_factory=list)
    
    # Métricas básicas
    accuracy: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    
    # NC1: Activation Collapse
    Sw_invSb: List[float] = field(default_factory=list)
    
    # NC2: Equinorm
    norm_M_CoV: List[float] = field(default_factory=list)
    norm_W_CoV: List[float] = field(default_factory=list)
    
    # NC2: Equiangularity
    cos_M: List[float] = field(default_factory=list)
    cos_W: List[float] = field(default_factory=list)
    
    # NC3: Self-Duality
    W_M_dist: List[float] = field(default_factory=list)
    
    # NC4: NCC Convergence
    NCC_mismatch: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List]:
        """Converte para dicionário (para salvar em YAML/JSON)."""
        return {
            'epochs': self.epochs,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'nc1_Sw_invSb': self.Sw_invSb,
            'nc2_norm_M_CoV': self.norm_M_CoV,
            'nc2_norm_W_CoV': self.norm_W_CoV,
            'nc2_cos_M': self.cos_M,
            'nc2_cos_W': self.cos_W,
            'nc3_W_M_dist': self.W_M_dist,
            'nc4_NCC_mismatch': self.NCC_mismatch
        }
    
    def summary(self) -> str:
        """Retorna resumo das métricas finais."""
        if not self.epochs:
            return "Nenhuma métrica registrada."
        
        return f"""
Neural Collapse Summary (Epoch {self.epochs[-1]})
{'='*50}
Accuracy:     {self.accuracy[-1]*100:.2f}%
Loss:         {self.loss[-1]:.4f}

NC1 (Sw/Sb):  {self.Sw_invSb[-1]:.4f}
NC2 (M CoV):  {self.norm_M_CoV[-1]:.4f}
NC2 (W CoV):  {self.norm_W_CoV[-1]:.4f}
NC2 (cos M):  {self.cos_M[-1]:.4f}
NC2 (cos W):  {self.cos_W[-1]:.4f}
NC3 (W-M):    {self.W_M_dist[-1]:.4f}
NC4 (NCC):    {self.NCC_mismatch[-1]:.4f}
{'='*50}
"""


# ==============================================================================
# 2. Feature Extractor com Hook
# ==============================================================================

class FeatureExtractor:
    """
    Extrai features da penúltima camada usando forward hook.
    
    O hook captura a ENTRADA da camada FC (features antes da classificação).
    """
    
    def __init__(self):
        self.features: Optional[torch.Tensor] = None
        self.hook_handle = None
    
    def hook(self, module, input, output):
        """Forward hook que captura a entrada do módulo."""
        self.features = input[0].clone()
    
    def register(self, module: torch.nn.Module):
        """Registra o hook no módulo."""
        self.hook_handle = module.register_forward_hook(self.hook)
    
    def remove(self):
        """Remove o hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def find_classifier_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Encontra a camada classificadora (última camada linear) do modelo.
    
    Args:
        model: Modelo PyTorch
        
    Returns:
        Camada classificadora (nn.Linear)
        
    Raises:
        ValueError: Se não encontrar a camada
    """
    # Nomes comuns para a camada classificadora
    classifier_names = ['fc', 'linear', 'classifier', 'head', 'output']
    
    for name in classifier_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, torch.nn.Linear):
                return layer
    
    # Tentar encontrar a última camada Linear
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((name, module))
    
    if linear_layers:
        return linear_layers[-1][1]
    
    raise ValueError(
        "Não foi possível encontrar a camada classificadora. "
        "O modelo deve ter um atributo 'fc', 'linear', 'classifier', ou similar."
    )


# ==============================================================================
# 3. Função principal de análise NC
# ==============================================================================

def analyze_neural_collapse(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Calcula todas as métricas de Neural Collapse para um modelo.
    
    Implementa análise em dois passes:
    - Pass 1: Calcula médias de classe e loss
    - Pass 2: Calcula covariância intra-classe, accuracy e NCC match
    
    Args:
        model: Modelo PyTorch (deve ter camada 'fc' ou 'linear')
        loader: DataLoader com os dados
        device: 'cuda' ou 'cpu'
        num_classes: Número de classes (C)
        verbose: Se True, mostra barras de progresso
        
    Returns:
        Dicionário com todas as métricas NC
    """
    model.eval()
    C = num_classes
    
    # Setup feature extraction
    feature_extractor = FeatureExtractor()
    classifier = find_classifier_layer(model)
    feature_extractor.register(classifier)
    
    # Inicialização
    N = [0 for _ in range(C)]  # Contagem por classe
    mean = [0 for _ in range(C)]  # Médias de classe
    Sw = None  # Covariância intra-classe (inicializada depois)
    
    loss = 0.0
    net_correct = 0
    NCC_match_net = 0
    total_samples = 0
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # ======================== PASS 1: Médias de classe ========================
    iterator = tqdm(loader, desc='Pass 1 (Mean)', leave=False) if verbose else loader
    
    with torch.no_grad():
        for data, target in iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            h = feature_extractor.features.view(data.shape[0], -1)
            
            # Inicializar Sw com dimensão correta
            if Sw is None:
                feature_dim = h.shape[1]
                Sw = torch.zeros(feature_dim, feature_dim, device=device)
                mean = [torch.zeros(feature_dim, device=device) for _ in range(C)]
            
            loss += criterion(output, target).item()
            
            for c in range(C):
                idxs = (target == c).nonzero(as_tuple=True)[0]
                if len(idxs) == 0:
                    continue
                
                h_c = h[idxs, :]
                mean[c] = mean[c] + torch.sum(h_c, dim=0)
                N[c] += h_c.shape[0]
    
    # Normalizar médias
    for c in range(C):
        if N[c] > 0:
            mean[c] = mean[c] / N[c]
        else:
            # Classe sem amostras - manter zero
            pass
    
    # Matriz de médias: feature_dim x C
    M = torch.stack(mean).T
    total_N = sum(N)
    loss /= total_N
    
    # ======================== PASS 2: Covariância e métricas ========================
    iterator = tqdm(loader, desc='Pass 2 (Cov)', leave=False) if verbose else loader
    
    with torch.no_grad():
        for data, target in iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            h = feature_extractor.features.view(data.shape[0], -1)
            
            for c in range(C):
                idxs = (target == c).nonzero(as_tuple=True)[0]
                if len(idxs) == 0:
                    continue
                
                h_c = h[idxs, :]
                
                # Within-class covariance: Sw += sum((h - mean)(h - mean)^T)
                z = h_c - mean[c].unsqueeze(0)  # B x feature_dim
                Sw += z.T @ z  # feature_dim x feature_dim
                
                # Network accuracy
                net_pred = torch.argmax(output[idxs, :], dim=1)
                net_correct += (net_pred == target[idxs]).sum().item()
                
                # NCC (Nearest Class Center) prediction
                # Distância de cada amostra para cada média de classe
                NCC_scores = torch.cdist(h_c, M.T)  # B x C
                NCC_pred = torch.argmin(NCC_scores, dim=1)
                NCC_match_net += (NCC_pred == net_pred).sum().item()
                
                total_samples += len(idxs)
    
    # Normalizar covariância
    Sw /= total_samples
    
    # Remover hook
    feature_extractor.remove()
    
    # ======================== Calcular métricas NC ========================
    
    # Média global
    muG = torch.mean(M, dim=1, keepdim=True)  # feature_dim x 1
    
    # Médias centradas (M - média global)
    M_ = M - muG  # feature_dim x C
    
    # Between-class covariance: Sb = (1/C) * M_ @ M_^T
    Sb = (M_ @ M_.T) / C  # feature_dim x feature_dim
    
    # Pesos do classificador
    W = classifier.weight.data  # C x feature_dim
    
    # Normas das médias e pesos
    M_norms = torch.norm(M_, dim=0)  # C normas
    W_norms = torch.norm(W.T, dim=0)  # C normas (W^T tem shape feature_dim x C)
    
    metrics = {}
    
    # ----- Métricas básicas -----
    metrics['accuracy'] = net_correct / total_samples
    metrics['loss'] = loss
    
    # ----- NC1: Activation Collapse -----
    # tr{Sw @ Sb^-1} - deve tender a 0
    Sw_np = Sw.cpu().numpy()
    Sb_np = Sb.cpu().numpy()
    
    try:
        # SVD truncada para inversa pseudo de Sb
        k = min(C - 1, Sb_np.shape[0] - 1, 50)  # Limitar k para estabilidade
        if k > 0:
            eigvec, eigval, _ = svds(Sb_np, k=k)
            # Filtrar eigenvalues muito pequenos
            mask = eigval > 1e-10
            if mask.sum() > 0:
                inv_Sb = eigvec[:, mask] @ np.diag(eigval[mask] ** (-1)) @ eigvec[:, mask].T
                metrics['Sw_invSb'] = np.trace(Sw_np @ inv_Sb)
            else:
                metrics['Sw_invSb'] = np.nan
        else:
            metrics['Sw_invSb'] = np.nan
    except Exception as e:
        print(f"Warning: NC1 calculation failed: {e}")
        metrics['Sw_invSb'] = np.nan
    
    # ----- NC2: Equinorm (Coeficiente de Variação das normas) -----
    # Deve tender a 0 (todas as normas iguais)
    metrics['norm_M_CoV'] = (torch.std(M_norms) / (torch.mean(M_norms) + 1e-10)).item()
    metrics['norm_W_CoV'] = (torch.std(W_norms) / (torch.mean(W_norms) + 1e-10)).item()
    
    # ----- NC2: Equiangularity (Mutual Coherence) -----
    # Deve tender a 0 (ângulos iguais entre vetores)
    def compute_coherence(V: torch.Tensor, num_classes: int) -> float:
        """
        Calcula mutual coherence.
        Para ETF (Equiangular Tight Frame), cos(angle) = -1/(C-1)
        """
        # Normalizar colunas
        V_norm = V / (torch.norm(V, dim=0, keepdim=True) + 1e-10)
        
        # Matriz de Gram (cossenos)
        G = V_norm.T @ V_norm  # C x C
        
        # Adicionar 1/(C-1) - para ETF ideal, isso zera os off-diagonais
        G = G + torch.ones((num_classes, num_classes), device=V.device) / (num_classes - 1)
        
        # Remover diagonal
        G = G - torch.diag(torch.diag(G))
        
        # Média do valor absoluto (L1 norm normalizada)
        return torch.norm(G, p=1).item() / (num_classes * (num_classes - 1))
    
    metrics['cos_M'] = compute_coherence(M_, C)
    metrics['cos_W'] = compute_coherence(W.T, C)
    
    # ----- NC3: Self-Duality -----
    # ||W^T - M_||^2 normalizado - deve tender a 0
    normalized_M = M_ / (torch.norm(M_, 'fro') + 1e-10)
    normalized_W = W.T / (torch.norm(W.T, 'fro') + 1e-10)
    metrics['W_M_dist'] = (torch.norm(normalized_W - normalized_M) ** 2).item()
    
    # ----- NC4: Convergence to NCC -----
    # Proporção de predições que diferem do NCC - deve tender a 0
    metrics['NCC_mismatch'] = 1 - NCC_match_net / total_samples
    
    return metrics


# ==============================================================================
# 4. Carregamento de checkpoints e análise
# ==============================================================================

def load_checkpoints_and_analyze(
    checkpoint_dir: str,
    model_class: Callable,
    loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    checkpoint_pattern: str = '*.pth',
    epoch_regex: str = r'epoch(\d+)',
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> NCMetricsTracker:
    """
    Carrega todos os checkpoints e calcula métricas NC para cada um.
    
    Args:
        checkpoint_dir: Diretório com os checkpoints
        model_class: Classe do modelo (ex: ResNet18)
        loader: DataLoader (preferencialmente train_loader)
        device: 'cuda' ou 'cpu'
        num_classes: Número de classes
        checkpoint_pattern: Padrão glob para encontrar checkpoints
        epoch_regex: Regex para extrair número da epoch do nome do arquivo
        model_kwargs: Argumentos adicionais para o construtor do modelo
        verbose: Se True, mostra progresso
        
    Returns:
        NCMetricsTracker com todas as métricas
    """
    tracker = NCMetricsTracker()
    
    if model_kwargs is None:
        model_kwargs = {}
    
    # Encontrar checkpoints
    pattern = os.path.join(checkpoint_dir, checkpoint_pattern)
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print(f"⚠️ Nenhum checkpoint encontrado em: {pattern}")
        return tracker
    
    # Função para extrair epoch
    def get_epoch(path: str) -> int:
        match = re.search(epoch_regex, os.path.basename(path))
        if match:
            return int(match.group(1))
        return 0
    
    # Ordenar por epoch
    checkpoints = sorted(checkpoints, key=get_epoch)
    
    if verbose:
        print(f"📁 Encontrados {len(checkpoints)} checkpoints")
        print(f"   Epochs: {[get_epoch(c) for c in checkpoints]}")
    
    # Analisar cada checkpoint
    iterator = tqdm(checkpoints, desc='Analyzing checkpoints') if verbose else checkpoints
    
    for ckpt_path in iterator:
        epoch = get_epoch(ckpt_path)
        
        try:
            # Criar modelo fresh
            model = model_class(num_classes=num_classes, **model_kwargs)
            
            # Carregar checkpoint
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # Suportar diferentes formatos de checkpoint
            if isinstance(ckpt, dict):
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                elif 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'])
                else:
                    # Assumir que o dict é o próprio state_dict
                    model.load_state_dict(ckpt)
            else:
                model.load_state_dict(ckpt)
            
            model = model.to(device)
            model.eval()
            
            # Calcular métricas NC
            metrics = analyze_neural_collapse(
                model, loader, device, num_classes, verbose=False
            )
            
            # Armazenar
            tracker.epochs.append(epoch)
            tracker.accuracy.append(metrics['accuracy'])
            tracker.loss.append(metrics['loss'])
            tracker.Sw_invSb.append(metrics['Sw_invSb'])
            tracker.norm_M_CoV.append(metrics['norm_M_CoV'])
            tracker.norm_W_CoV.append(metrics['norm_W_CoV'])
            tracker.cos_M.append(metrics['cos_M'])
            tracker.cos_W.append(metrics['cos_W'])
            tracker.W_M_dist.append(metrics['W_M_dist'])
            tracker.NCC_mismatch.append(metrics['NCC_mismatch'])
            
            # Liberar memória
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"⚠️ Erro ao processar {ckpt_path}: {e}")
            continue
    
    if verbose:
        print(f"\n✅ Análise concluída: {len(tracker.epochs)} checkpoints processados")
    
    return tracker


# ==============================================================================
# 5. Plotting
# ==============================================================================

def plot_nc_evolution(
    tracker: NCMetricsTracker,
    save_dir: Optional[str] = None,
    figsize: tuple = (16, 10),
    dpi: int = 150
) -> plt.Figure:
    """
    Plota a evolução das métricas NC ao longo do treinamento.
    
    Args:
        tracker: NCMetricsTracker com as métricas
        save_dir: Diretório para salvar figura (opcional)
        figsize: Tamanho da figura
        dpi: Resolução para salvar
        
    Returns:
        Objeto Figure do matplotlib
    """
    if not tracker.epochs:
        print("⚠️ Nenhuma métrica para plotar")
        return None
    
    epochs = tracker.epochs
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle('Neural Collapse Evolution During Training', fontsize=14, fontweight='bold')
    
    # Cores consistentes
    color_loss = '#E74C3C'
    color_acc = '#27AE60'
    color_nc1 = '#3498DB'
    color_M = '#9B59B6'
    color_W = '#E67E22'
    color_nc3 = '#1ABC9C'
    color_nc4 = '#34495E'
    
    # ----- 1. Loss -----
    ax = axes[0, 0]
    ax.semilogy(epochs, tracker.loss, color=color_loss, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # ----- 2. Error Rate -----
    ax = axes[0, 1]
    error_rate = [100 * (1 - a) for a in tracker.accuracy]
    ax.plot(epochs, error_rate, color=color_acc, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (%)')
    ax.set_title('Training Error')
    ax.grid(True, alpha=0.3)
    
    # ----- 3. NC1: Activation Collapse -----
    ax = axes[0, 2]
    # Filtrar NaN
    valid_nc1 = [(e, v) for e, v in zip(epochs, tracker.Sw_invSb) if not np.isnan(v)]
    if valid_nc1:
        e_nc1, v_nc1 = zip(*valid_nc1)
        ax.semilogy(e_nc1, v_nc1, color=color_nc1, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Tr{Sw Sb⁻¹}')
    ax.set_title('NC1: Activation Collapse')
    ax.grid(True, alpha=0.3)
    
    # ----- 4. NC2: Equinorm -----
    ax = axes[0, 3]
    ax.plot(epochs, tracker.norm_M_CoV, color=color_M, linewidth=2, 
            marker='o', markersize=4, label='Class Means')
    ax.plot(epochs, tracker.norm_W_CoV, color=color_W, linewidth=2, 
            marker='s', markersize=4, label='Classifiers')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Std/Mean of Norms')
    ax.set_title('NC2: Equinorm')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ----- 5. NC2: Equiangularity -----
    ax = axes[1, 0]
    ax.plot(epochs, tracker.cos_M, color=color_M, linewidth=2, 
            marker='o', markersize=4, label='Class Means')
    ax.plot(epochs, tracker.cos_W, color=color_W, linewidth=2, 
            marker='s', markersize=4, label='Classifiers')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg|Cos + 1/(C-1)|')
    ax.set_title('NC2: Maximal Equiangularity')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ----- 6. NC3: Self-Duality -----
    ax = axes[1, 1]
    ax.semilogy(epochs, tracker.W_M_dist, color=color_nc3, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||W^T - M||²')
    ax.set_title('NC3: Self Duality')
    ax.grid(True, alpha=0.3)
    
    # ----- 7. NC4: NCC Convergence -----
    ax = axes[1, 2]
    ax.semilogy(epochs, tracker.NCC_mismatch, color=color_nc4, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Proportion Mismatch')
    ax.set_title('NC4: Convergence to NCC')
    ax.grid(True, alpha=0.3)
    
    # ----- 8. Summary Panel -----
    ax = axes[1, 3]
    ax.axis('off')
    
    # Criar texto de resumo
    summary_lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  Final Epoch: {epochs[-1]}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"  Accuracy:    {tracker.accuracy[-1]*100:.2f}%",
        f"  Loss:        {tracker.loss[-1]:.4f}",
        "",
        "  Neural Collapse Metrics:",
        f"  ─────────────────────────",
        f"  NC1 (Sw/Sb): {tracker.Sw_invSb[-1]:.4f}" if not np.isnan(tracker.Sw_invSb[-1]) else "  NC1 (Sw/Sb): N/A",
        f"  NC2 (M CoV): {tracker.norm_M_CoV[-1]:.4f}",
        f"  NC2 (W CoV): {tracker.norm_W_CoV[-1]:.4f}",
        f"  NC2 (cos M): {tracker.cos_M[-1]:.4f}",
        f"  NC2 (cos W): {tracker.cos_W[-1]:.4f}",
        f"  NC3 (W≈M):   {tracker.W_M_dist[-1]:.4f}",
        f"  NC4 (NCC):   {tracker.NCC_mismatch[-1]:.4f}",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  Ideal NC: all metrics → 0",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    
    summary_text = "\n".join(summary_lines)
    ax.text(0.05, 0.5, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', 
                     edgecolor='#DEE2E6', alpha=0.9))
    
    plt.tight_layout()
    
    # Salvar se diretório especificado
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'nc_evolution.png')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 Figura salva: {save_path}")
    
    return fig


def plot_nc_individual(
    tracker: NCMetricsTracker,
    save_dir: Optional[str] = None,
    dpi: int = 150
) -> Dict[str, plt.Figure]:
    """
    Plota cada métrica NC em figura separada (para relatórios).
    
    Args:
        tracker: NCMetricsTracker com as métricas
        save_dir: Diretório para salvar figuras
        dpi: Resolução
        
    Returns:
        Dicionário com figuras
    """
    if not tracker.epochs:
        print("⚠️ Nenhuma métrica para plotar")
        return {}
    
    epochs = tracker.epochs
    figures = {}
    
    plots_config = [
        ('nc1_activation_collapse', 'NC1: Activation Collapse', 
         'Tr{Sw Sb⁻¹}', tracker.Sw_invSb, True),
        ('nc2_equinorm_M', 'NC2: Equinorm (Class Means)', 
         'Std/Mean of Norms', tracker.norm_M_CoV, False),
        ('nc2_equinorm_W', 'NC2: Equinorm (Classifiers)', 
         'Std/Mean of Norms', tracker.norm_W_CoV, False),
        ('nc2_equiangular_M', 'NC2: Equiangularity (Class Means)', 
         'Avg|Cos + 1/(C-1)|', tracker.cos_M, False),
        ('nc2_equiangular_W', 'NC2: Equiangularity (Classifiers)', 
         'Avg|Cos + 1/(C-1)|', tracker.cos_W, False),
        ('nc3_self_duality', 'NC3: Self Duality', 
         '||W^T - M||²', tracker.W_M_dist, True),
        ('nc4_ncc_convergence', 'NC4: Convergence to NCC', 
         'Proportion Mismatch', tracker.NCC_mismatch, True),
    ]
    
    for name, title, ylabel, data, use_log in plots_config:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Filtrar NaN
        valid_data = [(e, v) for e, v in zip(epochs, data) if not np.isnan(v)]
        if valid_data:
            e_valid, v_valid = zip(*valid_data)
            if use_log:
                ax.semilogy(e_valid, v_valid, 'b-o', linewidth=2, markersize=5)
            else:
                ax.plot(e_valid, v_valid, 'b-o', linewidth=2, markersize=5)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures[name] = fig
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{name}.png')
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    if save_dir:
        print(f"📊 {len(figures)} figuras salvas em: {save_dir}")
    
    return figures


# ==============================================================================
# 6. Utilitários
# ==============================================================================

def save_metrics_yaml(tracker: NCMetricsTracker, filepath: str):
    """Salva métricas em arquivo YAML."""
    import yaml
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Converter para tipos nativos Python (YAML não aceita numpy)
    metrics_dict = {}
    for key, value in tracker.to_dict().items():
        if isinstance(value, list):
            metrics_dict[key] = [
                float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v)
                else (None if isinstance(v, float) and np.isnan(v) else v)
                for v in value
            ]
        else:
            metrics_dict[key] = value
    
    with open(filepath, 'w') as f:
        yaml.dump(metrics_dict, f, default_flow_style=False)
    
    print(f"💾 Métricas salvas: {filepath}")


def save_metrics_json(tracker: NCMetricsTracker, filepath: str):
    """Salva métricas em arquivo JSON."""
    import json
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    metrics_dict = {}
    for key, value in tracker.to_dict().items():
        if isinstance(value, list):
            metrics_dict[key] = [
                float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v)
                else (None if isinstance(v, float) and np.isnan(v) else v)
                for v in value
            ]
        else:
            metrics_dict[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"💾 Métricas salvas: {filepath}")


# ==============================================================================
# 7. Exemplo de uso (main)
# ==============================================================================

if __name__ == '__main__':
    """
    Exemplo de uso do módulo.
    
    Para usar no seu projeto:
    
    ```python
    from nc_analysis import load_checkpoints_and_analyze, plot_nc_evolution, save_metrics_yaml
    from src.models import ResNet18
    
    # Carregar dados
    train_loader, val_loader, test_loader = get_cifar100_loaders(...)
    
    # Analisar checkpoints
    tracker = load_checkpoints_and_analyze(
        checkpoint_dir='/content/drive/MyDrive/Colab Notebooks/OOD/Checkpoints',
        model_class=ResNet18,
        loader=train_loader,
        device='cuda',
        num_classes=100,
        checkpoint_pattern='resnet18_cifar100_*.pth',
        epoch_regex=r'epoch(\d+)'
    )
    
    # Plotar evolução
    plot_nc_evolution(tracker, save_dir='results/figures/neural_collapse')
    
    # Salvar métricas
    save_metrics_yaml(tracker, 'results/metrics/nc_evolution.yaml')
    
    # Ver resumo
    print(tracker.summary())
    ```
    """
    print(__doc__)
    print("\nPara usar este módulo, importe as funções necessárias:")
    print("  from nc_analysis import load_checkpoints_and_analyze, plot_nc_evolution")

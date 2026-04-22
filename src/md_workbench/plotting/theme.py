from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from ..config import PlotStyleConfig


def _sans_serif_stack(style: PlotStyleConfig) -> list[str]:
    candidates = [
        style.font_family,
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
    ]
    ordered = []
    for item in candidates:
        clean = str(item).strip()
        if clean and clean not in ordered:
            ordered.append(clean)
    return ordered


@contextmanager
def publication_style(style: PlotStyleConfig):
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _sans_serif_stack(style),
            "font.size": style.tick_size,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.label_size,
            "legend.fontsize": style.legend_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "axes.linewidth": style.axes_line_width,
            "grid.alpha": style.grid_alpha,
            "grid.linewidth": 0.7,
            "grid.color": style.grid_color,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "axes.edgecolor": style.spine_color,
            "xtick.color": style.spine_color,
            "ytick.color": style.spine_color,
            "axes.labelcolor": style.spine_color,
            "text.color": style.spine_color,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        yield


def apply_minor_ticks(ax, style: PlotStyleConfig):
    if style.use_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="minor", direction="out", length=2.5, width=max(0.6, style.axes_line_width * 0.8))


def finalize_axes(ax, style: PlotStyleConfig, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=8, weight="semibold")
    if style.show_grid:
        ax.grid(True, axis="both", linestyle="-", linewidth=0.7)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(style.axes_line_width)
        ax.spines[side].set_color(style.spine_color)
    ax.tick_params(direction="out", length=4.0, width=style.axes_line_width)
    apply_minor_ticks(ax, style)


@dataclass(frozen=True)
class FigureNoteTemplate:
    name: str
    explanation: str
    caption_example: str
    data_patterns: tuple[str, ...] = ()


FIGURE_NOTE_TEMPLATES: dict[str, FigureNoteTemplate] = {
    "rmsd_timeseries": FigureNoteTemplate(
        name="蛋白与配体 RMSD 时间序列",
        explanation="该图展示蛋白主链与配体重原子相对参考构象的 RMSD 随时间变化，用于评估体系整体结构稳定性以及配体构象漂移程度。",
        caption_example="蛋白主链与配体重原子相对参考构象的 RMSD 随时间变化。较小且趋于平稳的 RMSD 常用于支持体系进入相对稳定采样阶段，但仍需结合其他结构指标综合判断。",
        data_patterns=("rmsd_timeseries.csv",),
    ),
    "min_distance_timeseries": FigureNoteTemplate(
        name="配体-蛋白最小距离时间序列",
        explanation="该图展示配体与蛋白之间最小重原子距离随时间的变化，可用于跟踪配体是否维持在结合口袋附近以及是否出现明显脱离事件。",
        caption_example="配体与蛋白最小重原子距离随时间变化。持续较小的距离通常支持配体维持口袋邻近构象，而突然升高可能提示接触减弱或部分解离。",
        data_patterns=("min_distance_timeseries.csv",),
    ),
    "rmsf_ca": FigureNoteTemplate(
        name="蛋白 Cα RMSF 剖面",
        explanation="该图展示蛋白各残基 Cα 原子的均方根涨落，用于识别柔性较高或较稳定的结构区段。",
        caption_example="蛋白各残基 Cα RMSF 剖面。峰值较高的区域通常代表局部柔性较强，需结合结构位置和功能位点进一步解释。",
        data_patterns=("rmsf_ca.csv",),
    ),
    "temperature_density": FigureNoteTemplate(
        name="温度与密度稳定性时间序列",
        explanation="该图展示生产阶段温度和密度随时间的变化，用于评估恒温恒压条件下体系热力学稳定性与体相平衡情况。",
        caption_example="生产阶段温度与密度随时间变化。温度围绕设定值波动且密度逐步稳定，可作为体系达到稳定采样状态的辅助证据。",
        data_patterns=("md_log_parsed.csv",),
    ),
    "energy": FigureNoteTemplate(
        name="势能与总能时间序列",
        explanation="该图展示生产阶段势能和总能随时间的变化，用于监测体系能量收敛行为以及是否存在异常漂移。",
        caption_example="生产阶段势能与总能随时间变化。能量曲线若无系统性发散，通常支持积分过程稳定；若出现持续漂移，则需检查参数化与采样设置。",
        data_patterns=("md_log_parsed.csv",),
    ),
    "interaction_counts_timeseries": FigureNoteTemplate(
        name="相互作用计数时间序列",
        explanation="该图展示接触、氢键和盐桥等相互作用数量随时间的变化，用于观察结合界面在模拟过程中的相互作用保持情况。",
        caption_example="结合界面接触、氢键和盐桥数量随时间变化。较稳定的相互作用计数通常支持配体结合模式在采样期间保持一致。",
        data_patterns=("interaction_counts_timeseries.csv",),
    ),
    "radius_of_gyration": FigureNoteTemplate(
        name="回转半径时间序列",
        explanation="该图展示蛋白回转半径随时间的变化，用于评估整体紧致性是否发生明显变化。",
        caption_example="蛋白回转半径随时间变化。回转半径的平稳波动通常支持整体折叠状态保持稳定，而持续升高可能提示构象松散化。",
        data_patterns=("shape_timeseries.csv",),
    ),
    "sasa_buried_surface": FigureNoteTemplate(
        name="SASA 与埋藏表面积时间序列",
        explanation="该图展示复合物溶剂可及表面积和埋藏表面积随时间的变化，用于评估配体结合界面的暴露程度与埋藏程度。",
        caption_example="复合物溶剂可及表面积与埋藏表面积随时间变化。埋藏表面积较大且稳定通常支持较持续的界面接触。",
        data_patterns=("shape_timeseries.csv",),
    ),
    "ligand_pose_metrics": FigureNoteTemplate(
        name="配体构象姿态指标时间序列",
        explanation="该图展示配体-口袋质心距离和配体取向角度随时间的变化，用于评估配体结合姿态是否稳定。",
        caption_example="配体-口袋质心距离与配体取向角随时间变化。若两类指标均在有限范围内波动，通常说明配体姿态相对稳定。",
        data_patterns=("ligand_pose_metrics.csv",),
    ),
    "ligand_torsions": FigureNoteTemplate(
        name="配体可旋转键二面角时间序列",
        explanation="该图展示配体关键可旋转键二面角随时间的变化，用于分析内部构象自由度和主要构象状态。",
        caption_example="配体关键二面角随时间变化。二面角在少数稳定区间内切换通常提示配体存在有限构象子状态，而频繁大幅跳变则说明内部柔性较高。",
        data_patterns=("ligand_torsions.csv",),
    ),
    "dssp_fractions": FigureNoteTemplate(
        name="二级结构组分时间序列",
        explanation="该图展示螺旋、折叠和无规卷曲等二级结构组分比例随时间的变化，用于评估蛋白整体二级结构保持情况。",
        caption_example="蛋白二级结构组分比例随时间变化。若主要组分占比保持稳定，通常说明整体二级结构未出现显著重排。",
        data_patterns=("dssp_fractions_timeseries.csv",),
    ),
    "dssp_residue_occupancy": FigureNoteTemplate(
        name="残基二级结构占据热图",
        explanation="该图展示各残基处于螺旋、折叠或无规卷曲状态的占据比例，用于识别局部二级结构稳定区和易重排区。",
        caption_example="各残基二级结构占据热图。高占据比例的结构状态代表该残基在采样中更稳定地维持对应二级结构。",
        data_patterns=("dssp_residue_occupancy.csv",),
    ),
    "representative_snapshots": FigureNoteTemplate(
        name="代表性构象快照",
        explanation="该图展示从轨迹中抽取的代表性结构快照，用于直观比较蛋白骨架与配体在不同时间点的构象变化。",
        caption_example="轨迹代表性结构快照。不同时间点快照可用于直观比较结合位点形貌和配体姿态是否发生明显变化。",
        data_patterns=("snapshots/snapshot_manifest.csv", "snapshot_manifest.csv"),
    ),
    "waterbridge_counts_timeseries": FigureNoteTemplate(
        name="水桥数量时间序列",
        explanation="该图展示蛋白-水-配体桥连水分子数量随时间的变化，用于评估水介导相互作用的持续性。",
        caption_example="桥连水分子数量随时间变化。若水桥数量长期非零且波动有限，通常说明水介导相互作用在结合界面中具有持续作用。",
        data_patterns=("waterbridge_counts_timeseries.csv",),
    ),
    "waterbridge_residue_occupancy_top20": FigureNoteTemplate(
        name="关键残基水桥占据排序图",
        explanation="该图展示与配体形成水桥频率最高的残基及其占据比例，用于识别水介导结合网络中的关键位点。",
        caption_example="关键残基水桥占据排序图。占据比例较高的残基通常是水介导相互作用网络中的重要节点。",
        data_patterns=("waterbridge_residue_occupancy_combined.csv", "summary_per_replica.csv"),
    ),
    "waterbridge_count_combined": FigureNoteTemplate(
        name="跨重复水桥数量汇总时间序列",
        explanation="该图展示多个重复中桥连水分子数量的平均趋势及离散程度，用于评估水桥网络的重复性。",
        caption_example="多重复桥连水分子数量的平均趋势及离散程度。重复间曲线一致性较高通常支持水桥网络具有一定可重复性。",
        data_patterns=("waterbridge_count_combined.csv",),
    ),
    "waterbridge_count_replot": FigureNoteTemplate(
        name="水桥数量重绘图",
        explanation="该图根据已汇总的多重复水桥数量数据重新绘制平均趋势，用于与其他综合图版保持统一风格。",
        caption_example="多重复桥连水分子数量重绘图。该图以汇总数据为基础，用于在统一排版下展示平均趋势及其波动范围。",
        data_patterns=("waterbridge_count_combined.csv",),
    ),
    "rmsd_combined": FigureNoteTemplate(
        name="跨重复 RMSD 汇总图",
        explanation="该图展示多个重复中蛋白主链与配体重原子 RMSD 的平均趋势和重复间离散程度，用于评估构象稳定性的重复性。",
        caption_example="多重复蛋白主链与配体重原子 RMSD 汇总图。平均曲线及其波动范围可用于比较结构稳定性在重复之间的一致性。",
        data_patterns=("rmsd_combined.csv",),
    ),
    "rmsd_replot_protein": FigureNoteTemplate(
        name="蛋白 RMSD 重绘图",
        explanation="该图基于汇总后的蛋白主链 RMSD 数据重绘平均趋势及波动范围，用于标准化展示蛋白稳定性。",
        caption_example="蛋白主链 RMSD 重绘图。该图由汇总 RMSD 数据重新组织而成，用于清晰展示平均趋势及标准差范围。",
        data_patterns=("rmsd_combined.csv",),
    ),
    "rmsd_replot_ligand": FigureNoteTemplate(
        name="配体 RMSD 重绘图",
        explanation="该图基于汇总后的配体重原子 RMSD 数据重绘平均趋势及波动范围，用于标准化展示配体姿态稳定性。",
        caption_example="配体重原子 RMSD 重绘图。该图由汇总 RMSD 数据重新组织而成，用于清晰展示平均趋势及标准差范围。",
        data_patterns=("rmsd_combined.csv",),
    ),
    "min_distance_combined": FigureNoteTemplate(
        name="跨重复最小距离汇总图",
        explanation="该图展示多个重复中配体与蛋白最小重原子距离的平均趋势及离散程度，用于评估结合接触是否稳定。",
        caption_example="多重复配体-蛋白最小距离汇总图。距离曲线若在重复间保持一致，通常支持结合界面接触具有可重复性。",
        data_patterns=("min_distance_combined.csv",),
    ),
    "rmsf_ca_combined": FigureNoteTemplate(
        name="跨重复 Cα RMSF 汇总图",
        explanation="该图展示多个重复综合得到的残基 Cα RMSF 均值和标准差，用于识别稳定柔性特征。",
        caption_example="多重复蛋白 Cα RMSF 汇总图。均值反映平均柔性水平，标准差反映不同重复间柔性估计的一致性。",
        data_patterns=("rmsf_ca_combined.csv",),
    ),
    "contact_occupancy_top20": FigureNoteTemplate(
        name="关键接触残基占据排序图",
        explanation="该图展示与配体形成接触频率最高的残基及其占据比例，并用圆点面积编码平均最小距离，用于同时识别接触频率和空间贴近程度均较突出的热点残基。",
        caption_example="关键接触残基占据排序图。接触占据比例较高且平均最小距离较短的残基通常代表更稳定、更贴近配体的界面接触热点。",
        data_patterns=("contact_occupancy_combined.csv", "contact_occupancy_distance_summary.csv"),
    ),
    "hbond_residue_occupancy_top20": FigureNoteTemplate(
        name="关键氢键残基占据排序图",
        explanation="该图展示与配体形成氢键频率最高的残基及其占据比例，用于识别主要极性相互作用位点。",
        caption_example="关键氢键残基占据排序图。氢键占据比例较高的残基通常是维持结合特异性的关键极性位点。",
        data_patterns=("hbond_residue_occupancy_combined.csv",),
    ),
    "salt_bridge_residue_occupancy": FigureNoteTemplate(
        name="盐桥残基占据排序图",
        explanation="该图展示与配体形成盐桥频率最高的残基及其占据比例，用于识别主要静电锚定位点。",
        caption_example="盐桥残基占据排序图。高占据盐桥通常提示对应残基在界面静电稳定性中具有较重要作用。",
        data_patterns=("salt_bridge_residue_occupancy_combined.csv",),
    ),
    "key_contact_distance_traces": FigureNoteTemplate(
        name="关键残基距离轨迹图",
        explanation="该图展示若干关键残基与配体之间平均距离随时间的变化，用于比较不同热点残基的接触保持情况。",
        caption_example="关键残基-配体距离轨迹图。持续较短且波动有限的距离通常说明该残基与配体保持较稳定接触。",
        data_patterns=("key_contact_distance_traces.csv",),
    ),
    "contact_count_combined": FigureNoteTemplate(
        name="跨重复接触数汇总图",
        explanation="该图展示多个重复中配体-蛋白接触数的平均趋势及其波动范围，用于评估界面接触网络的稳定性。",
        caption_example="多重复接触数汇总图。平均接触数较高且重复间波动较小通常支持界面接触网络较稳定。",
        data_patterns=("contact_count_combined.csv",),
    ),
    "hbond_count_combined": FigureNoteTemplate(
        name="跨重复氢键数汇总图",
        explanation="该图展示多个重复中氢键数量的平均趋势及其波动范围，用于评估极性结合网络的稳定性。",
        caption_example="多重复氢键数汇总图。氢键数量在重复间较一致时，通常说明极性相互作用模式具有较好可重复性。",
        data_patterns=("hbond_count_combined.csv",),
    ),
    "salt_bridge_count_combined": FigureNoteTemplate(
        name="跨重复盐桥数汇总图",
        explanation="该图展示多个重复中盐桥数量的平均趋势及其波动范围，用于评估静电相互作用的持续性。",
        caption_example="多重复盐桥数汇总图。盐桥数量长期保持非零且重复间一致时，通常支持稳定的静电锚定作用。",
        data_patterns=("salt_bridge_count_combined.csv",),
    ),
    "radius_of_gyration_combined": FigureNoteTemplate(
        name="跨重复回转半径汇总图",
        explanation="该图展示多个重复中蛋白回转半径的平均趋势及其离散程度，用于评估整体紧致性的一致性。",
        caption_example="多重复蛋白回转半径汇总图。回转半径在重复间保持接近通常说明整体构象紧致性具有较好重复性。",
        data_patterns=("radius_of_gyration_combined.csv",),
    ),
    "buried_surface_combined": FigureNoteTemplate(
        name="跨重复埋藏表面积汇总图",
        explanation="该图展示多个重复中配体结合导致的埋藏表面积平均趋势及其离散程度，用于评估界面埋藏稳定性。",
        caption_example="多重复埋藏表面积汇总图。埋藏表面积较大且重复间一致通常支持结合界面较稳定。",
        data_patterns=("buried_surface_combined.csv",),
    ),
    "ligand_com_distance_combined": FigureNoteTemplate(
        name="跨重复质心距离汇总图",
        explanation="该图展示多个重复中配体-口袋质心距离的平均趋势及其离散程度，用于评估配体定位稳定性。",
        caption_example="多重复配体-口袋质心距离汇总图。距离较小且重复间变化一致通常支持配体持续定位于结合位点附近。",
        data_patterns=("ligand_com_distance_combined.csv",),
    ),
    "ligand_orientation_angle_combined": FigureNoteTemplate(
        name="跨重复配体取向角汇总图",
        explanation="该图展示多个重复中配体取向角的平均趋势及其离散程度，用于评估姿态方向是否保持一致。",
        caption_example="多重复配体取向角汇总图。取向角若集中在有限范围内并在重复间保持一致，通常说明配体姿态较稳定。",
        data_patterns=("ligand_orientation_angle_combined.csv",),
    ),
    "sasa_components_combined": FigureNoteTemplate(
        name="跨重复 SASA 组分汇总图",
        explanation="该图展示复合物、蛋白和配体的平均溶剂可及表面积随时间的变化，用于分析溶剂暴露特征。",
        caption_example="复合物、蛋白和配体平均 SASA 随时间变化。不同组分的暴露面积变化可用于辅助解释界面埋藏和构象紧致性变化。",
        data_patterns=("sasa_components_combined.csv",),
    ),
    "dssp_residue_occupancy_combined": FigureNoteTemplate(
        name="跨重复残基二级结构占据热图",
        explanation="该图展示多个重复综合得到的残基二级结构占据比例，用于识别重复间一致的稳定结构区。",
        caption_example="多重复残基二级结构占据热图。高占据区域代表在不同重复中较稳定保持同一二级结构状态的残基。",
        data_patterns=("dssp_residue_occupancy_combined.csv",),
    ),
    "dssp_fractions_combined": FigureNoteTemplate(
        name="跨重复二级结构组分汇总图",
        explanation="该图展示多个重复平均后的二级结构组分比例随时间变化，用于评估蛋白整体二级结构保持的一致性。",
        caption_example="多重复平均二级结构组分比例随时间变化。若主要组分占比在不同重复中保持一致，通常支持整体二级结构较稳定。",
        data_patterns=("../replica_*/dssp_fractions_timeseries.csv",),
    ),
    "contact_replicate_heatmap": FigureNoteTemplate(
        name="接触占据跨重复热图",
        explanation="该图展示关键接触残基在不同重复中的接触占据比例，用于比较接触热点的重复性。",
        caption_example="关键接触残基在不同重复中的接触占据热图。若同一残基在多数重复中占据较高，通常说明其为稳健的接触热点。",
        data_patterns=("contact_occupancy_combined.csv",),
    ),
    "interaction_fingerprint_heatmap": FigureNoteTemplate(
        name="界面相互作用指纹热图",
        explanation="该图综合展示关键残基在接触、氢键和盐桥三类相互作用中的占据水平，用于识别多模式界面热点。",
        caption_example="界面相互作用指纹热图。兼具高接触、高氢键或高盐桥占据的残基通常是更值得关注的综合界面热点。",
        data_patterns=("contact_occupancy_combined.csv", "hbond_residue_occupancy_combined.csv", "salt_bridge_residue_occupancy_combined.csv"),
    ),
    "convergence_block_heatmap": FigureNoteTemplate(
        name="收敛区块标准化偏离热图",
        explanation="该图展示多个指标在不同轨迹区块中的标准化偏离程度，其中每一行都在对应指标内部完成标准化，用于比较区块间相对漂移而不混淆不同量纲。",
        caption_example="多指标收敛区块标准化偏离热图。颜色表示相对各指标自身均值的偏离程度，后期区块若更接近零偏离，通常支持采样逐步趋于稳定。",
        data_patterns=("convergence_block_means_combined.csv", "convergence_block_zscores_combined.csv"),
    ),
    "replicate_consistency_boxplot": FigureNoteTemplate(
        name="标准化重复一致性箱线图",
        explanation="该图展示多个重复在若干核心指标上的标准化偏离分布，其中每个指标先按重复间均值和标准差归一化，用于避免不同量纲直接混比。",
        caption_example="核心结构指标的标准化重复一致性箱线图。不同指标的分布若大多围绕零偏离，通常说明重复间不存在明显系统性偏移。",
        data_patterns=("summary_per_replica.csv", "replicate_consistency_zscores.csv"),
    ),
    "replicate_consistency_zscore_heatmap": FigureNoteTemplate(
        name="重复一致性标准化热图",
        explanation="该图展示各重复在若干核心指标上的标准化偏离程度，用于快速识别是否存在某一重复在特定指标上系统性偏离其余重复。",
        caption_example="重复一致性标准化热图。颜色表示相对重复均值的偏离强度；若各单元格多接近零偏离，则通常支持重复间一致性较好。",
        data_patterns=("summary_per_replica.csv", "replicate_consistency_zscores.csv"),
    ),
    "explained_variance_ratio": FigureNoteTemplate(
        name="降维解释方差图",
        explanation="该图展示 PCA 各主成分的解释方差比例，用于评估前几个主成分能够捕获多少总体构象变化。",
        caption_example="PCA 各主成分解释方差比例图。前若干主成分累计解释方差较高时，通常说明主要构象变化可被低维表示较好概括。",
        data_patterns=("explained_variance_ratio.csv",),
    ),
    "free_energy_landscape_pc1_pc2": FigureNoteTemplate(
        name="PC1-PC2 自由能地形图",
        explanation="该图基于 PC1 和 PC2 上的构象分布估算相对自由能地形，颜色和等高线共同表示相对自由能，用于识别主要低能构象区域和潜在构象转变通道。",
        caption_example="基于 PC1 和 PC2 投影的相对自由能地形图。相对自由能以全局最低点归一为 0，标记的低自由能盆地通常对应更常见的构象亚稳态。",
        data_patterns=("free_energy_landscape_pc1_pc2.csv", "free_energy_landscape_pc1_pc2.json"),
    ),
    "free_energy_landscape_pc1_pc2_3d": FigureNoteTemplate(
        name="PC1-PC2 三维自由能地形图",
        explanation="该图以三维曲面形式展示 PC1 和 PC2 上估算得到的相对自由能，其中 z 轴高度对应相对自由能，适合直观看低能盆地的深浅和相对分离。",
        caption_example="PC1 和 PC2 投影上的三维相对自由能地形图。z 轴表示相对自由能，较低的谷底通常对应更常见的构象亚稳态。",
        data_patterns=("free_energy_landscape_pc1_pc2.csv", "free_energy_landscape_pc1_pc2.json"),
    ),
    "pc1_pc2_scatter": FigureNoteTemplate(
        name="PCA 投影散点图",
        explanation="该图展示不同重复在前两个主成分空间中的投影分布，用于比较重复之间采样到的构象空间是否一致。",
        caption_example="不同重复在 PC1-PC2 空间中的构象投影散点图。不同重复的投影若高度重叠，通常支持采样到相近的主要构象空间。",
        data_patterns=("../per_replica_assignments/*_pc.csv",),
    ),
    "singular_values": FigureNoteTemplate(
        name="tICA 奇异值图",
        explanation="该图展示 tICA 各慢模态的奇异值，用于评估慢动力学信号在不同时间独立成分中的分布。",
        caption_example="tICA 奇异值图。较大的奇异值通常意味着对应慢模态对体系慢动力学行为的贡献更显著。",
        data_patterns=("singular_values.csv",),
    ),
    "free_energy_landscape_tic1_tic2": FigureNoteTemplate(
        name="tIC1-tIC2 自由能地形图",
        explanation="该图基于 tIC1 和 tIC2 上的构象分布估算相对自由能地形，颜色和等高线共同表示相对自由能，用于识别慢动力学主导下的主要亚稳态。",
        caption_example="基于 tIC1 和 tIC2 投影的相对自由能地形图。相对自由能以全局最低点归一为 0，标记的低自由能盆地通常对应慢动力学空间中的主要亚稳态。",
        data_patterns=("free_energy_landscape_tic1_tic2.csv", "free_energy_landscape_tic1_tic2.json"),
    ),
    "free_energy_landscape_tic1_tic2_3d": FigureNoteTemplate(
        name="tIC1-tIC2 三维自由能地形图",
        explanation="该图以三维曲面形式展示 tIC1 和 tIC2 上估算得到的相对自由能，其中 z 轴高度对应相对自由能，适合直观看慢动力学空间中低能盆地的深浅和分离关系。",
        caption_example="tIC1 和 tIC2 投影上的三维相对自由能地形图。z 轴表示相对自由能，较低的谷底通常对应慢动力学空间中的主要亚稳态。",
        data_patterns=("free_energy_landscape_tic1_tic2.csv", "free_energy_landscape_tic1_tic2.json"),
    ),
    "tic1_tic2_scatter": FigureNoteTemplate(
        name="tICA 投影散点图",
        explanation="该图展示不同重复在前两个时间独立成分空间中的投影分布，用于比较慢动力学采样的一致性。",
        caption_example="不同重复在 tIC1-tIC2 空间中的构象投影散点图。不同重复若覆盖相似区域，通常说明慢动力学采样结果较一致。",
        data_patterns=("../per_replica_assignments/*_tic.csv",),
    ),
    "cluster_population_overall": FigureNoteTemplate(
        name="聚类总体占比图",
        explanation="该图展示全部采样帧在各聚类状态中的占比，用于识别最主要的构象状态。",
        caption_example="聚类总体占比图。占比更高的聚类通常代表体系更常访问的主要构象状态。",
        data_patterns=("cluster_population_overall.csv",),
    ),
    "clusters_tic1_tic2": FigureNoteTemplate(
        name="聚类结果散点图",
        explanation="该图展示 tICA 空间中的聚类分配结果和聚类中心，用于观察不同构象状态在慢模态空间中的分离程度。",
        caption_example="tICA 空间中的聚类结果散点图。聚类中心和不同状态的分离程度可用于辅助判断状态划分是否清晰。",
        data_patterns=("cluster_centers.csv", "../per_replica_assignments/*_tic.csv", "../per_replica_assignments/*_cluster_assignment.csv"),
    ),
    "state_population_by_replica": FigureNoteTemplate(
        name="各重复状态占比热图",
        explanation="该图展示各重复在不同聚类状态中的帧占比，用于比较不同重复对主要状态的采样偏好。",
        caption_example="各重复状态占比热图。若主要状态在多数重复中均有较高占比，通常说明状态分布具有较好重复性。",
        data_patterns=("cluster_population_per_replica.csv",),
    ),
    "representative_state_snapshots": FigureNoteTemplate(
        name="代表性状态结构快照",
        explanation="该图展示主要聚类状态的代表性结构快照，用于直观比较不同亚稳态下蛋白与配体的结构差异。",
        caption_example="主要聚类状态的代表性结构快照。不同状态之间的结构差异可用于辅助解释自由能地形和状态转变关系。",
        data_patterns=("representative_frames.csv", "cluster_*_*.pdb"),
    ),
    "stationary_distribution": FigureNoteTemplate(
        name="MSM 平稳分布图",
        explanation="该图展示 Markov 状态模型中各状态的平稳概率，用于评估长期极限下各状态的相对权重。",
        caption_example="MSM 平稳分布图。平稳概率更高的状态通常代表长期采样下更稳定或更常访问的构象状态。",
        data_patterns=("stationary_distribution.csv",),
    ),
    "transition_matrix_heatmap": FigureNoteTemplate(
        name="MSM 转移矩阵热图",
        explanation="该图展示状态之间的转移概率矩阵，用于分析状态间互相转换的可能性和动力学连通性。",
        caption_example="MSM 转移矩阵热图。高转移概率元素提示对应状态对之间存在更频繁的动力学交换。",
        data_patterns=("transition_matrix.csv",),
    ),
    "state_network": FigureNoteTemplate(
        name="MSM 状态网络图",
        explanation="该图以网络形式展示状态间平衡转移通量和各状态的相对权重，用于直观理解构象状态之间的动力学连通结构。",
        caption_example="MSM 状态网络图。节点大小反映平稳概率，边的粗细反映平衡转移通量 pi_i P_ij，边标签给出对应转移概率 P_ij。",
        data_patterns=("transition_matrix.csv", "stationary_distribution.csv", "equilibrium_transition_flux.csv"),
    ),
    "implied_timescales_single_lag": FigureNoteTemplate(
        name="单滞后时间尺度图",
        explanation="该图展示在给定 MSM 滞后时间下不同动力学过程的隐含时间尺度，用于衡量慢过程分离程度。",
        caption_example="给定滞后时间下的 MSM 隐含时间尺度图。较长的时间尺度通常对应更慢的构象转换过程。",
        data_patterns=("implied_timescales_single_lag.csv",),
    ),
    "implied_timescales_lag_scan": FigureNoteTemplate(
        name="隐含时间尺度滞后扫描图",
        explanation="该图展示不同滞后时间下的隐含时间尺度变化，用于评估 MSM 构建是否达到近似马尔可夫行为。",
        caption_example="不同滞后时间下的隐含时间尺度扫描图。时间尺度曲线随滞后时间趋于平台常被用作 MSM 合理性的辅助判断依据。",
        data_patterns=("implied_timescales_lag_scan.csv",),
    ),
    "mmgbsa_summary": FigureNoteTemplate(
        name="MM/GBSA 能量分项汇总图",
        explanation="该图展示 MM/GBSA 各能量分项的平均贡献及其离散程度，用于分析不同能量项对结合自由能的相对影响。",
        caption_example="MM/GBSA 能量分项汇总图。不同能量分项的相对大小可用于辅助解释结合自由能的主要驱动因素，但应结合方法近似条件谨慎解读。",
        data_patterns=("mmpbsa_summary.csv", "mmpbsa_summary_parsed.csv"),
    ),
    "mmgbsa_per_frame": FigureNoteTemplate(
        name="MM/GBSA 按帧能量时间序列",
        explanation="该图展示 MM/GBSA 能量项在逐帧或逐时间点上的变化，用于评估结合自由能估计在采样过程中的波动。",
        caption_example="MM/GBSA 按帧能量时间序列图。曲线波动范围反映采样窗口内能量估计的不确定性与时间依赖性。",
        data_patterns=("mmpbsa_FINAL_RESULTS.csv",),
    ),
    "mmgbsa_per_residue": FigureNoteTemplate(
        name="MM/GBSA 逐残基分解图",
        explanation="该图展示各残基对结合自由能的逐残基贡献，用于识别潜在的关键有利或不利位点。",
        caption_example="MM/GBSA 逐残基贡献排序图。绝对值较大的残基通常是值得优先关注的关键贡献位点，但需注意分解结果本身具有模型近似性。",
        data_patterns=("mmpbsa_DECOMP.csv",),
    ),
    "mmgbsa_delta_total_summary": FigureNoteTemplate(
        name="跨重复 MM/GBSA 总自由能汇总图",
        explanation="该图展示各重复的 MM/GBSA DELTA TOTAL 平均值及其标准差，用于比较不同重复的结合自由能估计是否一致。",
        caption_example="各重复 MM/GBSA DELTA TOTAL 平均值及标准差。重复间结果若接近，通常支持自由能估计在当前采样下具有一定一致性。",
        data_patterns=("mmgbsa_delta_total_summary.csv",),
    ),
    "mmgbsa_delta_total_per_frame": FigureNoteTemplate(
        name="跨重复 MM/GBSA DELTA TOTAL 按帧曲线",
        explanation="该图展示多个重复的 MM/GBSA DELTA TOTAL 按帧变化，用于比较不同重复在采样过程中的自由能波动模式。",
        caption_example="多个重复的 MM/GBSA DELTA TOTAL 按帧变化曲线。不同重复曲线的重叠程度可用于辅助判断自由能估计的一致性。",
        data_patterns=("../replica_*/mmpbsa_FINAL_RESULTS.csv",),
    ),
    "mmgbsa_delta_total_distribution": FigureNoteTemplate(
        name="跨重复 MM/GBSA DELTA TOTAL 分布图",
        explanation="该图展示多个重复中 MM/GBSA DELTA TOTAL 的按帧分布范围，用于比较不同重复自由能估计的离散程度。",
        caption_example="多个重复的 MM/GBSA DELTA TOTAL 按帧分布图。分布宽度越大，通常说明该重复内自由能估计波动越明显。",
        data_patterns=("../replica_*/mmpbsa_FINAL_RESULTS.csv",),
    ),
    "mmgbsa_per_residue_heatmap": FigureNoteTemplate(
        name="跨重复 MM/GBSA 逐残基热图",
        explanation="该图展示多个重复中关键残基逐残基贡献的分布，用于比较关键位点贡献在重复之间是否一致。",
        caption_example="跨重复 MM/GBSA 逐残基贡献热图。若同一残基在多个重复中均呈现相似贡献方向和幅度，通常说明其作用更稳健。",
        data_patterns=("../replica_*/mmpbsa_DECOMP.csv",),
    ),
    "mmgbsa_per_residue_top": FigureNoteTemplate(
        name="关键残基 MM/GBSA 逐残基排序图",
        explanation="该图展示绝对贡献最大的若干残基平均逐残基贡献，用于突出最可能影响结合自由能的关键位点。",
        caption_example="关键残基 MM/GBSA 逐残基贡献排序图。平均贡献绝对值较大的残基通常更值得优先结合结构信息进一步分析。",
        data_patterns=("../replica_*/mmpbsa_DECOMP.csv",),
    ),
}


def _humanize_stem(stem: str) -> str:
    text = stem.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else stem


def _tokenize_stem(stem: str) -> set[str]:
    pieces = {item for item in re.split(r"[^a-z0-9]+", stem.lower()) if item}
    expanded = set(pieces)
    for item in list(pieces):
        if item.startswith("pc"):
            expanded.add("pc")
        if item.startswith("tic"):
            expanded.add("tic")
        if item.startswith("mmgbsa"):
            expanded.add("mmgbsa")
    return expanded


def _collect_figure_context(fig) -> tuple[str, list[str], list[str]]:
    titles: list[str] = []
    axis_lines: list[str] = []
    legend_labels: list[str] = []

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        text = suptitle.get_text().strip()
        if text:
            titles.append(text)

    for idx, ax in enumerate(fig.axes, start=1):
        title = ax.get_title().strip()
        xlabel = ax.get_xlabel().strip()
        ylabel = ax.get_ylabel().strip()
        _, labels = ax.get_legend_handles_labels()
        if not any([title, xlabel, ylabel, labels]):
            continue
        if title:
            titles.append(title)
        parts = []
        if xlabel:
            parts.append(f"x轴为 {xlabel}")
        if ylabel:
            parts.append(f"y轴为 {ylabel}")
        if parts:
            axis_lines.append(f"面板 {idx}: " + "，".join(parts))
        for label in labels:
            clean = str(label).strip()
            if clean and not clean.startswith("_") and clean not in legend_labels:
                legend_labels.append(clean)
    return (titles[0] if titles else ""), axis_lines, legend_labels


def _candidate_data_directories(base_path: Path) -> list[Path]:
    candidates = [base_path.parent]
    parts = base_path.parts
    if "figures_combined" in parts:
        idx = parts.index("figures_combined")
        mapped = Path(*parts[:idx], "process_data", *parts[idx + 1 : -1])
        if mapped not in candidates:
            candidates.append(mapped)
    return candidates


def _resolve_data_patterns(search_dirs: list[Path], patterns: tuple[str, ...] | list[str]) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    for search_dir in search_dirs:
        for pattern in patterns:
            candidate_pattern = search_dir / pattern
            if any(token in pattern for token in "*?[]"):
                found = sorted(search_dir.glob(pattern))
            else:
                found = [candidate_pattern] if candidate_pattern.exists() else []
            for item in found:
                resolved = item.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    matches.append(resolved)
    return matches


def _guess_reproduction_data(base_path: Path, extra_patterns: tuple[str, ...] = ()) -> list[Path]:
    search_dirs = _candidate_data_directories(base_path)
    stem = base_path.stem
    matches: list[Path] = []
    seen: set[Path] = set()

    def add_path(path: Path) -> None:
        resolved = path.resolve()
        if path.exists() and resolved not in seen:
            seen.add(resolved)
            matches.append(resolved)

    for directory in search_dirs:
        for suffix in (".csv", ".json", ".txt"):
            same_name = directory / f"{stem}{suffix}"
            if same_name.exists():
                add_path(same_name)

    for item in _resolve_data_patterns(search_dirs, extra_patterns):
        add_path(item)

    if not matches:
        target_tokens = _tokenize_stem(stem)
        scored: list[tuple[int, Path]] = []
        for directory in search_dirs:
            for candidate in sorted(directory.glob("*.csv")) + sorted(directory.glob("*.json")):
                overlap = target_tokens & _tokenize_stem(candidate.stem)
                if overlap:
                    scored.append((len(overlap), candidate.resolve()))
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        for _, candidate in scored[:5]:
            if candidate not in seen:
                seen.add(candidate)
                matches.append(candidate)
    return matches


def _build_fallback_note(display_name: str) -> FigureNoteTemplate:
    return FigureNoteTemplate(
        name=display_name,
        explanation=f"该图展示“{display_name}”对应的分析结果。建议结合坐标定义、图例和原始数据文件共同解读。",
        caption_example=f"{display_name}。图中趋势和分布应结合原始数值结果及重复间一致性综合解读。",
        data_patterns=(),
    )


def _write_figure_note(fig, base_path: Path, formats: list[str]) -> Path:
    if fig is None:
        primary_title, axis_lines, legend_labels = "", [], []
    else:
        primary_title, axis_lines, legend_labels = _collect_figure_context(fig)
    template = FIGURE_NOTE_TEMPLATES.get(base_path.stem)
    if template is None:
        template = _build_fallback_note(primary_title or _humanize_stem(base_path.stem))
    image_files = [base_path.with_suffix(f".{fmt}").name for fmt in formats]
    data_files = _guess_reproduction_data(base_path, template.data_patterns)
    display_name = template.name
    explanation_parts = [template.explanation]
    if axis_lines:
        explanation_parts.append("坐标与面板定义: " + "；".join(axis_lines) + "。")
    if legend_labels:
        explanation_parts.append("图中主要数据系列包括: " + "、".join(legend_labels) + "。")
    explanation = " ".join(explanation_parts)

    lines = [
        f"图片名称: {display_name}",
        f"图片文件: {', '.join(image_files)}",
    ]
    if primary_title and primary_title != display_name:
        lines.append(f"图中标题: {primary_title}")
    lines.extend(
        [
        "",
        "图片解释说明:",
        explanation,
        "",
        "图注示例:",
        template.caption_example,
        "",
        "用于复现该图片的所用数据:",
        ]
    )
    if data_files:
        lines.extend(f"- {path}" for path in data_files)
    else:
        lines.append("- 未自动匹配到结构化数据文件，请结合该图所在目录的上游分析结果检查。")

    note_path = base_path.with_suffix(".txt")
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note_path


def write_figure_note(base_path: str | Path, formats: list[str] | None = None) -> Path:
    base_path = Path(base_path)
    if formats is None:
        formats = [
            suffix.lstrip(".")
            for suffix in (".png", ".svg", ".pdf")
            if base_path.with_suffix(suffix).exists()
        ]
    return _write_figure_note(None, base_path, list(formats))


def save_figure(fig, base_path: str | Path, style: PlotStyleConfig):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in style.formats:
        fig.savefig(
            base_path.with_suffix(f".{fmt}"),
            dpi=style.dpi,
            transparent=style.transparent_background,
        )
    _write_figure_note(fig, base_path, list(style.formats))
    plt.close(fig)

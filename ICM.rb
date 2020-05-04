require 'date'
require 'csv'

# 新建数组，存放random.csv中储存的随机数
random_num = Array.new
CSV.read('icm_model_data/valid_random_num.csv', headers:false).each do |row|
    random_num << row
end

# bat文件输入的Network和Run的命名随机数
input_num = ARGV[1]
model_name = ARGV[2]
temp_network_name = "#{model_name}_#{input_num}"

# 连接ICM数据库
db = WSApplication.open
group = db.model_object('>MODG~Graduate Project>MODG~run_batch')  # 读取存放run的模型库
leve_group = db.model_object('>MODG~Graduate Project>MODG~level')  # 读取存放level的模型库
pollution_group = db.model_object('>MODG~Graduate Project>MODG~pollution')  # 读取存放level的模型库
inflow_group = db.model_object('>MODG~Graduate Project>MODG~inflow')  # 读取存放level的模型库


# 删除未删除成功的run
begin

    old_run = db.model_object(">MODG~Graduate Project>MODG~run_batch>RUN~#{temp_network_name}")
    if !old_run.nil?
        old_run.bulk_delete
    end
    
    rescue RuntimeError
        puts "没有同名Run"
end


# 找到netwrok
net = db.model_object(">MODG~Graduate Project>NNET~#{model_name}")  # 存放Network的模型库
on = net.open  # 提取Network中的OpenNetwork才可以操作Scenario和执行Validaton

# 导入水位曲线
leve_group.import_new_model_object('Level', "icm_level_#{input_num}", 'CSV',
    "C:/Users/Carlisle/iCloudDrive/Graduate_Program/ICM-Delft3D/icm_model_data/icm_level.csv", 0)
icm_level = db.model_object(">MODG~Graduate Project>MODG~level>LEV~icm_level_#{input_num}")
icm_level_id = icm_level.id

# 导入污染物曲线和入流曲线
# 新建pollutograph的同时储存其id
pollutograph_array = Array.new
pollutograph_id = {}
inflow_array = Array.new
inflow_id = {}
random_num.each do |num|
    if num[0] != '-1'
        # 生成污染物曲线
        pollution_group.new_model_object("Pollutant Graph", "icm_pollution_#{num[0]}")
        pollutograph = db.model_object(">MODG~Graduate Project>MODG~pollution>PGR~icm_pollution_#{num[0]}")
        pollutograph_array << pollutograph
        pollutograph_id[num[0]] = pollutograph.id
        pollutograph.import_data("NHD", "C:/Users/Carlisle/iCloudDrive/Graduate_Program/ICM-Delft3D/icm_model_data/icm_pollution_#{num[0]}.nhd") # 未解决
        # 生成入流曲线
        inflow_group.import_new_model_object('Inflow', "icm_inflow_#{num[0]}", 'CSV',
            "C:/Users/Carlisle/iCloudDrive/Graduate_Program/ICM-Delft3D/icm_model_data/icm_inflow_#{num[0]}.csv", 0)
        inflow = db.model_object(">MODG~Graduate Project>MODG~inflow>INF~icm_inflow_#{num[0]}") 
        inflow_id[num[0]] = inflow.id
        inflow_array << inflow
        sleep 0.05 # 休息0.1秒
    end
end

# 从Run模板读取参数
run_template = ARGV[3]   # 从bat文件输入run模板的名字
runParms = {}
run_template = db.model_object(">MODG~Graduate Project>MODG~runs>RUN~#{run_template}") 
if !run_template.nil?
    db.list_read_write_run_fields.each do |p|
        if !run_template[p].nil?  # 如果参数不为nil
            runParms[p] = run_template[p]  # 读取参数
        end
    end
end

# 新建run
simsArray = Array.new  # 传入launch_sims的SimObject要用Array储存（参数格式要求）
runsArray = Array.new
simsNums = Array.new
random_num.each do |num|
    if num[0] != '-1'
        runParms['Level'] = icm_level_id
        runParms['Inflow'] = inflow_id[num[0]]
        runParms['Pollutant Graph'] = pollutograph_id[num[0]]
        run = group.new_run(num[0], net, nil, nil, "Base", runParms)
        # sim为run下面的子Object
        simsArray << run.children[0] 
        runsArray << run
        simsNums << num[0]
        sleep 0.1 # 休息0.1秒
    end
end

sleep 10 # 休息10秒，让数据库缓冲一下

# 运行模拟
WSApplication.connect_local_agent(1)
handles = WSApplication.launch_sims(simsArray, '.', false, 1, 0)

# 模型运行监测器
select_list = db.model_object(">MODG~Graduate Project>MODG~selection_list>SEL~monitors_new")
sim_status = Array.new(simsArray.size, 0)  # 模型导出与否的标签,0：未导出，1：已导出
sim_retry = Array.new(simsArray.size, 0)  # 模型失败重新运行的次数标签
sim_running = true
while sim_running
    sim_running = false # 先假设都跑完了，去验证是否跑完
    completed_sim = 0 # 已完成模拟的数量
    for i in 0...simsArray.size
        # 重新运行成功提示
        if sim_status[i] == 0 and simsArray[i].status == "Success" and sim_retry[i] > 0
            puts "#{simsNums[i]}重新运行成功"
        end
        # 导出结果        
        if sim_status[i] == 0 and simsArray[i].status == "Success"  # 如果运行成果且标记表示还未导出结果，则导出结果
            simsArray[i].results_csv_export_ex(select_list,[['Node',['depnod','mcnh4tot']], ['Link',['ds_flow', 'ds_mcnh4tot']]],
                "C:/Users/Carlisle/iCloudDrive/Graduate_Program/ICM-Delft3D/icm_result")
            sim_status[i] = 1  # 标记为导出
            completed_sim = completed_sim + 1
            puts "已完成#{completed_sim}个模拟"
            sleep 0.01
        end
        # 失败重新运行
        if simsArray[i].status == "Fail" and sim_retry[i] <= 3  # 失败且重试次数小于3
            puts "#{simsNums[i]}运行失败，重试次数#{sim_retry[i]}"
            # 重新运行模型
            failed_sim = Array.new # 记得要用Array装SimObject
            failed_sim << simsArray[i]
            handles = WSApplication.launch_sims(failed_sim, '.', false, 1, 0)
            sim_retry[i] = sim_retry[i] + 1  # 重试次数 + 1
            puts "#{simsNums[i]}重新运行"
        end
        # 判断模拟是否全部运行完
        if simsArray[i].status != "Success" or  sim_status[i] == 0  # 模拟未完成
            sim_running = true  # 有一个模型没跑完，都还在running
        end
    end
end
sleep 5
puts "模拟完成"
# 删除生成的临时文件
icm_level.bulk_delete

runsArray.each do |temp|
    temp.bulk_delete
end

pollutograph_array.each do |temp|
    temp.bulk_delete
end

inflow_array.each do |temp|
    temp.bulk_delete
end